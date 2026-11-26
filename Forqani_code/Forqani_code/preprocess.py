"""Steps 1 and 2 - image corrections before segmentation (helper functions).

Items 1-2 are called by step1_image_corrections.py (STEP 1); items 3-4 by
run_all_v3.py (STEP 2).

1. Constant-valued voxels: one exact non-zero value repeated ~450k times.
   * 'shell'   : connected to the brain/background boundary -> treated as non-tissue
   * 'interior': isolated voxels inside tissue -> in-painted from valid neighbours
2. Planar intensity artefacts: planes whose WM voxels deviate from the planes two
   away (robust outlier rule per axis; only sagittal i=165-166 in sub-405). The
   planes are replaced by interpolation between the nearest good planes.
3. Smooth regional intensity offset (right anterior-inferior brightening):
   estimated as a low-frequency ADDITIVE field from tissue-class residuals
   (the data showed all classes shifted by a similar amount, WM-GM difference
   preserved, so an additive model fits better than a multiplicative one).
4. Very bright voxels (arteries/sinuses, brighter than any WM): flagged so that
   they can be excluded from the white-matter mask.
"""
import numpy as np
from scipy import ndimage as ndi
from sklearn.mixture import GaussianMixture


def find_constant_value(data, min_count=1000):
    """Return the most frequent non-zero value if it is suspiciously frequent."""
    vals, counts = np.unique(data[data != 0], return_counts=True)
    i = np.argmax(counts)
    return (float(vals[i]), int(counts[i])) if counts[i] >= min_count else (None, 0)


def split_constant(data, value):
    const = data == np.float32(value)
    background = data == 0
    # shell = constant voxels whose 26-connected component touches the background
    lab, _ = ndi.label(const, structure=np.ones((3, 3, 3)))
    touching = np.unique(lab[ndi.binary_dilation(background, np.ones((3, 3, 3))) & const])
    shell = np.isin(lab, touching[touching > 0])
    interior = const & ~shell
    return shell, interior


def inpaint_median(data, holes, valid, max_iter=10):
    """Fill `holes` with the median of valid 3x3x3 neighbours, iterating inwards."""
    out = data.copy()
    todo = holes.copy()
    ok = valid & ~holes
    idx_all = np.argwhere(todo)
    for _ in range(max_iter):
        if not todo.any():
            break
        idx = np.argwhere(todo)
        filled = np.zeros(len(idx), bool)
        for n, (i, j, k) in enumerate(idx):
            sl = (slice(max(i - 1, 0), i + 2), slice(max(j - 1, 0), j + 2), slice(max(k - 1, 0), k + 2))
            nb = out[sl][ok[sl]]
            if nb.size >= 3:
                out[i, j, k] = np.median(nb)
                filled[n] = True
        f = idx[filled]
        ok[f[:, 0], f[:, 1], f[:, 2]] = True
        todo[f[:, 0], f[:, 1], f[:, 2]] = False
    return out, todo  # todo = voxels that could not be filled


def detect_artifact_planes(data, valid, wm_level=0.6, z=6.0, min_vox=3000):
    """Flag planes whose WM voxels are darker/brighter than the planes 2 away.

    Using only voxels whose neighbours (p-2, p+2) are WM-like avoids flagging
    anatomy (e.g. the CSF-filled interhemispheric fissure at the midline).
    """
    found = {}
    for axis in range(3):
        dd = np.moveaxis(data, axis, 0)
        vv = np.moveaxis(valid, axis, 0)
        ratios = {}
        for p in range(2, dd.shape[0] - 2):
            a, b = dd[p - 2], dd[p + 2]
            sel = vv[p] & vv[p - 2] & vv[p + 2] & (a > wm_level) & (b > wm_level)
            if sel.sum() >= min_vox:
                ratios[p] = np.median(dd[p][sel] / ((a[sel] + b[sel]) / 2))
        r = np.array(list(ratios.values()))
        med = np.median(r)
        mad = 1.4826 * np.median(np.abs(r - med))
        found[axis] = sorted(p for p, x in ratios.items() if abs(x - med) > z * mad)
    return found


def interpolate_planes(data, valid, planes):
    """Replace flagged planes by linear interpolation between the nearest good planes.

    A per-plane linear correction was tried first, but the distortion is not exactly
    linear (a ~2% residual remained in WM), so interpolation is used.
    Along the sagittal axis this is benign in the vermis: folia run left-right,
    so structures are continuous across the replaced ~1.1 mm.
    """
    out = data.copy()
    for axis, plist in planes.items():
        if not plist:
            continue
        dd = np.moveaxis(out, axis, 0)
        vv = np.moveaxis(valid, axis, 0)
        bad = set(plist)
        for p in plist:
            lo = max(q for q in range(p) if q not in bad)
            hi = min(q for q in range(p + 1, dd.shape[0]) if q not in bad)
            w = (p - lo) / (hi - lo)
            new = (1 - w) * dd[lo] + w * dd[hi]
            m = vv[p] & vv[lo] & vv[hi]
            dd[p][m] = new[m]
    return out


def fit_tissue_gmm(x, seed=0, n=300000):
    rng = np.random.default_rng(seed)
    x = x[rng.choice(len(x), min(n, len(x)), replace=False)].reshape(-1, 1)
    g = GaussianMixture(3, random_state=seed).fit(x)
    order = np.argsort(g.means_.ravel())  # CSF < GM < WM
    return g, order


def additive_bias_field(data, valid, voxel_mm, sigma_mm=15.0, factor=4, n_iter=3):
    """Low-frequency additive offset from GM/WM residuals (computed at 1/factor resolution)."""
    corr = data.copy()
    field_total = np.zeros_like(data)
    for _ in range(n_iter):
        g, order = fit_tissue_gmm(corr[valid])
        mu = g.means_.ravel()[order]
        x = corr[valid].reshape(-1, 1)
        post = g.predict_proba(x)[:, order]
        cls = post.argmax(1)
        conf = post.max(1) > 0.9
        resid = np.zeros(x.shape[0], np.float32)
        w = np.zeros(x.shape[0], np.float32)
        for c in (1, 2):  # GM and WM only (CSF class is mostly partial volume)
            sel = conf & (cls == c)
            resid[sel] = x[sel, 0] - mu[c]
            w[sel] = 1.0
        R = np.zeros_like(data)
        W = np.zeros_like(data)
        R[valid], W[valid] = resid * w, w
        # low resolution estimate
        Rs = R[::factor, ::factor, ::factor]
        Ws = W[::factor, ::factor, ::factor]
        s = sigma_mm / (voxel_mm * factor)
        F = ndi.gaussian_filter(Rs, s) / np.maximum(ndi.gaussian_filter(Ws, s), 1e-3)
        F[ndi.gaussian_filter(Ws, s) < 0.05] = 0
        zoom = [a / b for a, b in zip(data.shape, F.shape)]
        Ffull = ndi.zoom(F, zoom, order=1)[:data.shape[0], :data.shape[1], :data.shape[2]]
        corr[valid] -= Ffull[valid]
        field_total += Ffull
    return corr, field_total


def bright_voxels(data, valid, k_sd=4.0):
    """Voxels brighter than WM mean + k*SD (vessels, sinuses), slightly dilated."""
    g, order = fit_tissue_gmm(data[valid])
    mu_wm = g.means_.ravel()[order][2]
    sd_wm = np.sqrt(g.covariances_.ravel()[order][2])
    thr = mu_wm + k_sd * sd_wm
    bright = valid & (data > thr)
    bright = ndi.binary_dilation(bright, np.ones((3, 3, 3))) & valid
    return bright, float(thr)
