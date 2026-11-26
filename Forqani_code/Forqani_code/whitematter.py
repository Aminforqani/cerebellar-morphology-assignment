"""STEP 4 - cerebellar white-matter segmentation.

Mirrors the logic of the group's tissue step (fuzzy clustering + ridge filter for
thin WM branches), re-implemented for an MPRAGE-like contrast (WM = BRIGHT, so the
ridge polarity is inverted relative to their T1-map pipeline):

1. 3-class Gaussian mixture inside the cerebellum (vessels excluded) -> mu_CSF, mu_GM, mu_WM
2. Partial-volume WM fraction  f = (I - mu_GM) / (mu_WM - mu_GM), clipped to [0, 1]
   (two-compartment GM/WM mixing model; f is also saved as a soft WM map)
3. Hessian 'sheetness' for bright thin sheets (WM laminae inside folia are sheets in 3D)
4. Ridge-guided hysteresis: strong WM = f >= t_high; weak voxels (t_low <= f < t_high)
   are added only if they are sheet-like AND connected to strong WM
5. Remove vessel voxels, keep the component connected to the WM trunk, fill tiny holes.

Prediction tested in QC: with a GM/WM mixing model, a lamina of thickness t inside a
voxel of size h gives f ~ t/h, so a threshold at fraction t_high only keeps laminae
with t >~ t_high * h (0.5 * 0.55 mm ~ 0.28 mm). Distal branches should therefore
disappear first when the threshold is raised.
"""
import numpy as np
from scipy import ndimage as ndi
from sklearn.mixture import GaussianMixture


def bbox(mask, pad=4):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, mask.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def fit_classes(values, seed=0, n=300000):
    rng = np.random.default_rng(seed)
    x = values[rng.choice(len(values), min(n, len(values)), replace=False)].reshape(-1, 1)
    g = GaussianMixture(3, random_state=seed).fit(x)
    o = np.argsort(g.means_.ravel())
    return g.means_.ravel()[o], np.sqrt(g.covariances_.ravel()[o]), g.weights_[o]


def sheetness(img, sigma=0.7):
    """Bright-sheet measure from Hessian eigenvalues (|l1| >= |l2| >= |l3|).

    A bright sheet has one large negative eigenvalue (across the sheet) and two
    small ones (within the sheet): S = -l1 * exp(-(l2^2 + l3^2) / (2 (0.5 l1)^2)) for l1 < 0.
    """
    H = np.empty(img.shape + (3, 3), np.float32)
    for a in range(3):
        for b in range(a, 3):
            order = [0, 0, 0]
            order[a] += 1
            order[b] += 1
            H[..., a, b] = H[..., b, a] = ndi.gaussian_filter(img, sigma, order=order) * sigma ** 2
    ev = np.linalg.eigvalsh(H)  # ascending
    idx = np.argsort(-np.abs(ev), axis=-1)
    ev = np.take_along_axis(ev, idx, axis=-1)
    l1, l2, l3 = ev[..., 0], ev[..., 1], ev[..., 2]
    S = np.where(l1 < 0, -l1 * np.exp(-(l2 ** 2 + l3 ** 2) / (2 * (0.5 * l1) ** 2 + 1e-12)), 0)
    return S.astype(np.float32)


def denoise(data, mask, sigma_noise=None, h_factor=1.0):
    """Non-local means inside the cerebellar bounding box (stand-in for the group's
    LCPCA denoising, which needs the raw INV1/INV2 magnitude+phase data).

    Noise SD is estimated from neighbour differences in deep WM when not given.
    """
    from skimage.restoration import denoise_nl_means
    sl = bbox(mask, pad=6)
    D = data[sl]
    if sigma_noise is None:
        mu, _, _ = fit_classes(D[mask[sl]])
        deep = ndi.binary_erosion(mask[sl] & (D > (mu[1] + mu[2]) / 2), iterations=3)
        diff = (D[1:] - D[:-1])[deep[1:] & deep[:-1]]
        sigma_noise = 1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2)
    out = data.copy()
    out[sl] = denoise_nl_means(D, h=h_factor * sigma_noise, sigma=sigma_noise,
                               patch_size=3, patch_distance=3, fast_mode=True)
    return out, float(sigma_noise)


def segment(data, cb_mask, bright, t_high=0.5, t_low=0.3, sheet_pct=50, sigma=0.7,
            min_hole=50, use_ridge=True, brainstem=None, min_depth_mm=1.0, voxel_mm=0.55):
    """`brainstem` (optional) marks the peduncle side of the cerebellar mask: WM may
    touch that cut, but nowhere else may WM lie closer than `min_depth_mm` to the
    outer (pial/tentorial) surface - cortex always covers the WM there. This removes
    bright surface structures (veins, dura, edge artefacts) that mimic WM laminae."""
    sl = bbox(cb_mask)
    D = data[sl].astype(np.float32)
    M = cb_mask[sl]
    V = bright[sl]
    fit_vox = M & ~V
    mu, sd, w = fit_classes(D[fit_vox])
    f = np.clip((D - mu[1]) / (mu[2] - mu[1]), 0, 1) * M

    allowed = M & ~V
    if min_depth_mm > 0:
        outside = ~M if brainstem is None else (~M & ~brainstem[sl])
        depth = ndi.distance_transform_edt(~outside) * voxel_mm
        allowed &= depth >= min_depth_mm
    strong = (f >= t_high) & allowed
    weak = (f >= t_low) & allowed
    if use_ridge:
        S = sheetness(np.where(M, D, mu[0]), sigma)
        s_thr = np.percentile(S[weak & ~strong], sheet_pct) if (weak & ~strong).any() else 0
        weak &= (S >= s_thr) | strong
    else:
        S, s_thr = None, None
    # hysteresis: weak voxels connected (26-conn) to strong voxels
    lab, _ = ndi.label(weak, np.ones((3, 3, 3)))
    keep = np.unique(lab[strong])
    wm = np.isin(lab, keep[keep > 0])
    # keep the trunk-connected component
    lab, _ = ndi.label(wm, np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    trunk = lab == np.argmax(sizes)
    removed_islands = int(wm.sum() - trunk.sum())
    wm = trunk
    # fill small holes (vessels / noise inside WM), not large ones (e.g. dentate)
    holes = ndi.binary_fill_holes(wm) & ~wm
    hl, _ = ndi.label(holes)
    hs = np.bincount(hl.ravel())
    small = np.isin(hl, np.where((hs > 0) & (hs < min_hole))[0]) & (hl > 0)
    wm |= small

    out_wm = np.zeros(cb_mask.shape, bool)
    out_wm[sl] = wm
    out_f = np.zeros(cb_mask.shape, np.float32)
    out_f[sl] = f
    out_s = None
    if S is not None:
        out_s = np.zeros(cb_mask.shape, np.float32)
        out_s[sl] = S
    info = dict(mu=mu.round(4).tolist(), sd=sd.round(4).tolist(), weights=w.round(3).tolist(),
                t_high=t_high, t_low=t_low, sheet_threshold=None if s_thr is None else float(s_thr),
                removed_islands_vox=removed_islands, filled_small_holes_vox=int(small.sum()))
    return out_wm, out_f, out_s, info
