"""STEP 6 (optional cross-check) - compare the atlas-based cerebellum with the intensity-based one, remove the
atlas dentate/nuclei from the WM mask, and make a QC figure.

Usage:
  python step6_atlas_check.py <corrected_T1w> <outputs_dir> <atlas_out_dir>
"""
import os, sys, json
import numpy as np
import ants
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

t1_f, odir, adir = sys.argv[1:4]
t1 = ants.image_read(t1_f)
rd = lambda f: ants.image_read(f).numpy()
T = t1.numpy()
cb = rd(os.path.join(odir, "cerebellum_mask.nii.gz")) > 0.5
wm = rd(os.path.join(odir, "cerebellar_wm_mask.nii.gz")) > 0.5
L = rd(os.path.join(adir, "atlas_labels_in_subject.nii.gz")).round().astype(int)
acb = (L >= 1) & (L <= 34)
dent = (L == 29) | (L == 30)
nuc = (L >= 29) & (L <= 34)
ml = float(np.prod(t1.spacing)) / 1000
dice = lambda a, b: 2 * (a & b).sum() / (a.sum() + b.sum())
wm_nodent = wm & ~dent
res = {
    "intensity_cerebellum_mL": cb.sum() * ml, "atlas_cerebellum_mL": acb.sum() * ml,
    "dice_cerebellum": dice(cb, acb),
    "intensity_only_mL": (cb & ~acb).sum() * ml, "atlas_only_mL": (acb & ~cb).sum() * ml,
    "atlas_dentate_mL": dent.sum() * ml, "dentate_inside_WM_mask_mL": (dent & wm).sum() * ml,
    "pct_of_dentate_labelled_WM": 100 * (dent & wm).sum() / max(dent.sum(), 1),
    "WM_mL": wm.sum() * ml, "WM_minus_dentate_mL": wm_nodent.sum() * ml,
    "WM_minus_all_nuclei_mL": (wm & ~nuc).sum() * ml,
    "mean_T1_dentate": float(T[dent].mean()), "mean_T1_WM_outside_nuclei": float(T[wm & ~nuc].mean()),
}
json.dump(res, open(os.path.join(adir, "atlas_vs_intensity.json"), "w"), indent=2)
ants.image_write(t1.new_image_like(wm_nodent.astype("float32")),
                 os.path.join(adir, "cerebellar_wm_mask_minus_dentate.nii.gz"))
for k, v in res.items(): print(f"{k}: {v:.3f}")

# QC figure: three slices through the dentate centre
c = np.round(ndi.center_of_mass(dent)).astype(int) if dent.any() else np.round(ndi.center_of_mass(cb)).astype(int)
cR = np.round(ndi.center_of_mass(L == 30)).astype(int) if (L == 30).any() else c
fig, ax = plt.subplots(1, 3, figsize=(15, 5.5))
bb = np.argwhere(cb | acb); lo, hi = np.maximum(bb.min(0) - 10, 0), bb.max(0) + 10
views = [("sagittal (right dentate)", lambda a: a[cR[0] - lo[0], :, :]), ("coronal", lambda a: a[:, c[1] - lo[1], :]),
         ("axial", lambda a: a[:, :, c[2] - lo[2]])]
for a, (t, f) in zip(ax, views):
    sl = lambda x: f(x[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).T
    a.imshow(sl(T), cmap="gray", origin="lower", vmin=np.percentile(T[cb], 1), vmax=np.percentile(T[cb], 99.5))
    a.contour(sl(cb.astype(float)), [0.5], colors="yellow", linewidths=0.8)
    a.contour(sl(acb.astype(float)), [0.5], colors="cyan", linewidths=0.8)
    a.contour(sl(wm.astype(float)), [0.5], colors="red", linewidths=0.4)
    a.contour(sl(dent.astype(float)), [0.5], colors="magenta", linewidths=1.2)
    a.set_title(t); a.axis("off")
fig.suptitle("yellow: intensity-based cerebellum | cyan: atlas cerebellum (ANTs SyN, Diedrichsen 2009) | "
             "red: WM mask | magenta: atlas dentate", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(adir, "qc_atlas_overlay.png"), dpi=130)
