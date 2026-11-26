"""STEP 1 - image quality corrections (run on its own or from run_all_v3.py).

Two problems found when inspecting the image (Question 1):
  A. ~450,000 voxels (4.9%) share one exact value (0.424918).
     - those touching the outer edge of the brain mask = not tissue -> removed
     - those inside tissue = isolated holes -> filled with the median of neighbours
  B. Two sagittal planes (i = 165-166) have compressed contrast (WM ~10% darker).
     -> replaced by linear interpolation between the neighbouring good planes.

Usage:  python step1_image_corrections.py <input.nii.gz> <outdir>
Writes: step1_corrected.nii.gz, step1_excluded.nii.gz (1 = removed shell, 2 = filled hole),
        step1_report.json
"""
import json, os, sys
import numpy as np
import preprocess as P
from io_nifti import load, save


def correct(d):
    """Return the corrected image, the tissue mask and a small report."""
    value, count = P.find_constant_value(d)              # A. the repeated value
    shell, holes = P.split_constant(d, value)            #    edge vs interior
    brain = (d > 0) & ~shell                             #    tissue mask without the shell
    d1, unfilled = P.inpaint_median(d, holes, brain)     #    fill interior holes
    planes = P.detect_artifact_planes(d1, brain)         # B. find abnormal planes
    d2 = P.interpolate_planes(d1, brain, planes)         #    replace them
    rep = {"constant_value": float(value), "constant_count": int(count),
           "pct_of_nonzero": round(100 * count / (d > 0).sum(), 2),
           "shell_removed": int(shell.sum()), "holes_filled": int(holes.sum()),
           "artifact_planes": {f"axis{a}": [int(x) for x in p] for a, p in planes.items()}}
    return d2, brain, shell, holes, rep


if __name__ == "__main__":
    inp, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    img = load(inp)
    d2, brain, shell, holes, rep = correct(img.data)
    save(os.path.join(out, "step1_corrected.nii.gz"), np.where(brain, d2, 0), img)
    save(os.path.join(out, "step1_excluded.nii.gz"), shell + 2 * holes.astype(np.uint8), img, np.uint8)
    json.dump(rep, open(os.path.join(out, "step1_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=1))
