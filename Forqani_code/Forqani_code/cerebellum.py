"""STEP 3 - isolate the cerebellum from image intensities only (no atlas).

An atlas is used afterwards only as an independent cross-check (steps 5-6).
Note: the voxel-index parameters below (k_range, k_cerebrum, k_cb_marker,
j_cb_marker, crop, min_start) were set for sub-405 and would need adapting
(or an atlas-based starting mask) for other subjects.

The group's pipeline transfers an MNI cerebellar prior by non-linear registration
and then refines the boundary with a dura (dark sheet) detector. Here the same two
ideas are re-implemented without registration:

1. Markers
   * brainstem : an explicit geometric model (below)
   * cerebrum  : all brain tissue above `k_cerebrum` (well above the tentorium)
   * cerebellum: eroded tissue that is inferior (k <= k_cb_marker) and posterior
                 (j <= j_cb_marker) to the brainstem - the core of the hemispheres/vermis
2. Marker-based watershed on an 'elevation' image that is high on dark voxels and
   higher still on thin dark lines (black top-hat): region fronts therefore meet on
   the tentorium / dura, the analogue of the group's dura ridge filter.
   Where no dark boundary exists (cerebellar peduncles) the cut falls where the
   cerebellar and brainstem fronts meet - this is reported as a limitation.
3. Clean-up: small opening, largest connected component, hole filling.

Brainstem model: for every axial slice the brainstem cross-section is the most
anterior long run of tissue on the midline; it is modelled as an ellipse centred on
that run (AP semi-axis = half the run length, LR semi-axis = lr_factor x AP).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball
from skimage.segmentation import watershed

SIX = ndi.generate_binary_structure(3, 1)


def midline_index(brain, k_max=215, search=20, k_min=0):
    """Left-right midline (index along axis 0) from mirror symmetry of the brain
    mask below k_max (posterior fossa + lower cerebrum).

    The whole-brain centre of mass was 2.5 mm off the cerebellar midline in sub-405,
    which biased the brainstem model and the left/right volumes.
    """
    m = brain[:, :, k_min:k_max]
    c0 = ndi.center_of_mass(m)[0]
    idx = np.arange(m.shape[0])
    best = (c0, -1.0)
    for c in np.arange(c0 - search, c0 + search + 0.5, 0.5):
        mi = np.round(2 * c - idx).astype(int)
        ok = (mi >= 0) & (mi < m.shape[0])
        a, b = m[idx[ok]], m[mi[ok]]
        dice = 2 * (a & b).sum() / (a.sum() + b.sum())
        if dice > best[1]:
            best = (c, dice)
    return int(round(best[0]))


def brainstem_runs(data, brain, i_mid, dark_thr, k_range, min_len=12, min_start=120, halfwidth=2):
    runs = {}
    band = slice(i_mid - halfwidth, i_mid + halfwidth + 1)
    for k in range(*k_range):
        prof = np.where(brain[band, :, k], data[band, :, k], 0).mean(0)
        lab, n = ndi.label(prof > dark_thr)
        best = None
        for r in range(1, n + 1):
            js = np.where(lab == r)[0]
            if len(js) >= min_len and js[0] >= min_start and (best is None or js[-1] > best[1]):
                best = (int(js[0]), int(js[-1]))
        if best:
            runs[k] = best
    return runs


def brainstem_model(shape, runs, i_mid, lr_factor=1.4, min_semi=8, smooth=5):
    ks = np.array(sorted(runs))
    js = np.array([runs[k][0] for k in ks], float)
    je = np.array([runs[k][1] for k in ks], float)
    jc = ndi.median_filter((js + je) / 2, smooth, mode='nearest')
    a = np.maximum(ndi.median_filter((je - js) / 2, smooth, mode='nearest'), min_semi)
    mask = np.zeros(shape, bool)
    I, J = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    for k, c, s in zip(ks, jc, a):
        mask[:, :, k] = ((J - c) / s) ** 2 + ((I - i_mid) / (lr_factor * s)) ** 2 <= 1.0
    return mask, dict(k_first=int(ks[0]), k_last=int(ks[-1]), jc_median=float(np.median(jc)),
                      a_median=float(np.median(a)))


def isolate(data, brain, **kw):
    """Run the isolation twice: the second pass uses the midline re-estimated from
    the mirror symmetry of the first-pass cerebellum mask."""
    cb, brainstem, bs_model, info = _isolate(data, brain, i_mid=None, **kw)
    i_cb = midline_index(cb, k_max=brain.shape[2], search=10)
    info['i_mid_first_pass'] = info['i_mid']
    if abs(i_cb - info['i_mid']) > 1:
        cb, brainstem, bs_model, info2 = _isolate(data, brain, i_mid=i_cb, **kw)
        info2['i_mid_first_pass'] = info['i_mid']
        info = info2
    return cb, brainstem, bs_model, info


def _isolate(data, brain, i_mid=None, dark_thr=0.33, k_range=(95, 215), lr_factor=1.4,
            k_cerebrum=222, k_cb_marker=150, j_cb_marker=135, tophat_weight=2.0,
            tophat_radius=2, crop=((30, 300), (10, 280), (90, 240))):
    if i_mid is None:
        i_mid = midline_index(brain)
    runs = brainstem_runs(data, brain, i_mid, dark_thr, k_range)
    bs_model, bs_info = brainstem_model(data.shape, runs, i_mid, lr_factor)
    bs_model &= brain

    sl = tuple(slice(*c) for c in crop)
    D = np.where(brain[sl], data[sl], 0).astype(np.float32)
    B = brain[sl]
    K = np.arange(*crop[2])[None, None, :]
    J = np.arange(*crop[1])[None, :, None]

    tissue = B & (D > dark_thr)
    core = ndi.binary_erosion(tissue & (K <= k_cb_marker) & (J <= j_cb_marker) & ~bs_model[sl],
                              SIX, iterations=2)
    lab, _ = ndi.label(core, SIX)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    markers = np.zeros(D.shape, np.int32)
    markers[B & (K >= k_cerebrum)] = 1
    markers[ndi.binary_erosion(bs_model[sl], SIX)] = 2
    markers[lab == np.argmax(sizes)] = 3

    tophat = ndi.grey_closing(D, footprint=ball(tophat_radius)) - D  # thin dark lines
    elevation = ndi.gaussian_filter(1.0 - D + tophat_weight * tophat, 0.7)
    ws = watershed(elevation, markers, mask=B)

    cb = ws == 3
    raw_ml = cb.sum()
    cb = ndi.binary_opening(cb, SIX)
    lab, _ = ndi.label(cb, SIX)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    cb = lab == np.argmax(sizes)
    cb = ndi.binary_closing(cb, SIX) & B
    cb = ndi.binary_fill_holes(cb)

    out = np.zeros(brain.shape, bool)
    out[sl] = cb
    brainstem = np.zeros(brain.shape, bool)
    brainstem[sl] = ws == 2
    info = dict(i_mid=i_mid, brainstem_model=bs_info, watershed_voxels=int(raw_ml),
                final_voxels=int(out.sum()))
    return out, brainstem, bs_model, info
