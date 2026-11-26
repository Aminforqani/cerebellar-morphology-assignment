"""Quality control for the cerebellar WM segmentation.

Usage: python qc.py <input.nii[.gz]> <output_dir>   (after run_all_v3.py)

1. Threshold sensitivity: WM volume vs t_high, split by depth below the
   cerebellar surface -> tests the prediction that distal (superficial)
   branches are lost first when the threshold is raised.
2. Checks: vessel overlap, islands, left/right balance, the corrected artefact
   planes, WM touching the brainstem cut, dentate region.
3. Figures: example slices (raw / cerebellum mask / WM) and the sensitivity curve.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import whitematter as W
from io_nifti import load


def main(inp, outdir):
    raw = load(inp)
    vox = float(raw.voxel_size[0])
    ml = vox ** 3 / 1000
    corr = load(os.path.join(outdir, 'corrected_T1w.nii.gz')).data
    cb = load(os.path.join(outdir, 'cerebellum_mask.nii.gz')).data > 0
    wm = load(os.path.join(outdir, 'cerebellar_wm_mask.nii.gz')).data > 0
    excl = load(os.path.join(outdir, 'excluded_voxels.nii.gz')).data.astype(np.uint8)
    bright = (excl & 4) > 0
    d4 = np.load(os.path.join(outdir, '_d4_denoised.npy'))
    bsl = np.load(os.path.join(outdir, '_brainstem_label.npy'))
    rep = json.load(open(os.path.join(outdir, 'report.json')))
    q = {}

    # ---- 1. threshold sensitivity by depth ---------------------------------
    depth = ndi.distance_transform_edt(cb) * vox  # mm below the cerebellar surface
    bins = [(0, 2), (2, 4), (4, 7), (7, 100)]
    ts = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    curve = {}
    for t in ts:
        m, _, _, _ = W.segment(d4, cb, bright, t_high=t, t_low=max(t - 0.2, 0.1), brainstem=bsl, voxel_mm=vox)
        curve[t] = [float((m & (depth > a) & (depth <= b)).sum() * ml) for a, b in bins]
    q['sensitivity_ml_by_depth_mm'] = {'bins_mm': bins, 'curve': curve}
    ref = curve[0.3]
    q['retained_fraction_at_t0.7_vs_t0.3'] = [round(c / r, 3) if r else None
                                              for c, r in zip(curve[0.7], ref)]

    # ---- 2. checks ---------------------------------------------------------
    q['wm_vessel_overlap_vox'] = int((wm & bright).sum())
    q['vessel_vox_inside_cerebellum'] = int((bright & cb).sum())
    lab, n = ndi.label(wm, np.ones((3, 3, 3)))
    q['wm_components'] = int(n)
    i_mid = rep['cerebellum']['i_mid']
    r_vol = wm[:i_mid].sum() * ml
    l_vol = wm[i_mid + 1:].sum() * ml
    q['wm_ml_right_left'] = [round(r_vol, 2), round(l_vol, 2)]
    q['wm_asymmetry_index'] = round((r_vol - l_vol) / ((r_vol + l_vol) / 2), 3)
    q['wm_fraction_per_sagittal_plane_160_171'] = {
        int(i): round(float(wm[i][cb[i]].mean()), 3) for i in range(160, 172)}
    # WM voxels on the cerebellar mask boundary facing the brainstem (= peduncle cut)
    border = cb & ~ndi.binary_erosion(cb)
    q['wm_on_cerebellar_border_vox'] = int((wm & border).sum())
    mu = rep['white_matter']['mu']
    gm_like = cb & ~wm & (d4 > (mu[0] + mu[1]) / 2)
    q['gm_to_wm_ratio_excl_csf'] = round(float(gm_like.sum() / wm.sum()), 2)

    with open(os.path.join(outdir, 'qc.json'), 'w') as f:
        json.dump(q, f, indent=2)
    print(json.dumps(q, indent=1))

    # ---- 3. figures --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for b, (a, bb) in enumerate(bins):
        y = [curve[t][b] for t in ts]
        lbl = f'{a}-{bb} mm' if bb < 100 else f'>{a} mm'
        ax[0].plot(ts, y, 'o-', label=lbl)
        ax[1].plot(ts, np.array(y) / y[0], 'o-', label=lbl)
    ax[0].set(xlabel='WM threshold (fraction between GM and WM mean)', ylabel='WM volume (mL)',
              title='WM volume by depth below surface')
    ax[1].set(xlabel='WM threshold', ylabel='retained fraction (vs t=0.3)', title='Relative retention')
    ax[1].legend(title='depth')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'qc_threshold_by_depth.png'), dpi=120)

    d0 = raw.data
    views = [('Sagittal', 'i', 120), ('Coronal', 'j', 100), ('Axial', 'k', 150)]
    lim = {'i': ((40, 190), (100, 215)), 'j': ((60, 260), (100, 205)), 'k': ((60, 260), (40, 190))}
    fig, ax = plt.subplots(3, 3, figsize=(15, 13))
    for c, (name, axis, s) in enumerate(views):
        def cut(v):
            return {'i': v[s].T, 'j': v[:, s, :].T, 'k': v[:, :, s].T}[axis]
        img0, img = cut(d0), cut(corr)
        flip = axis != 'i'  # show coronal/axial in neurological convention (L on left)
        for r, (base, ov, title) in enumerate([(img0, None, 'Original'),
                                               (img, cut(cb), 'Corrected + cerebellum'),
                                               (img, cut(wm), 'Cerebellar WM')]):
            a = ax[r, c]
            b = base[:, ::-1] if flip else base
            a.imshow(b, cmap='gray', origin='lower', vmin=0.15, vmax=0.8, interpolation='nearest')
            if ov is not None:
                o = ov[:, ::-1] if flip else ov
                if r == 1:
                    a.contour(o, [0.5], colors='cyan', linewidths=0.8)
                else:
                    a.imshow(np.ma.masked_where(~o, o), cmap='autumn', alpha=0.5, origin='lower',
                             interpolation='nearest')
            xl, yl = lim[axis]
            if flip:
                xl = (d0.shape[0] - xl[1], d0.shape[0] - xl[0])
            a.set_xlim(*xl)
            a.set_ylim(*yl)
            a.set_xticks([])
            a.set_yticks([])
            if r == 0:
                a.set_title(f'{name} {axis}={s}')
            if c == 0:
                a.set_ylabel(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'qc_slices.png'), dpi=110)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
