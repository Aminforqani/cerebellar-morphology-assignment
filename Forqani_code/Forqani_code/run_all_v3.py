"""Cerebellar white-matter segmentation of a 0.55 mm MPRAGE-like MP2RAGE - version 3.

Same method and same results as version 2; report.json now also stores the
cerebellum and white-matter details that qc.py needs.

Usage:  python run_all_v3.py <input.nii[.gz]> <output_dir>

  STEP 1  step1_image_corrections.py  constant-value voxels + two abnormal sagittal planes
  STEP 2  preprocess.py               smooth regional brightness offset + very bright vessels
          (preprocess.py also holds the helper functions used by step 1)
  STEP 3  cerebellum.py               cut the cerebellum out (watershed, no atlas)
  STEP 4  whitematter.py              denoise, WM fraction, thin-sheet filter, thresholds
  (QC)    qc.py                       threshold-by-depth test and figures
  STEP 5  step5_atlas_registration.py optional cross-check: ANTs + Diedrichsen atlas
  STEP 6  step6_atlas_check.py        compare atlas vs our masks, dentate nucleus
"""
import json, os, sys, time
import numpy as np
import cerebellum
import preprocess as P
import whitematter as W
from io_nifti import load, save
from step1_image_corrections import correct


def main(inp, outdir):
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    img = load(inp)
    ml = float(img.voxel_size[0]) ** 3 / 1000.0          # mL per voxel

    # STEP 1 - image corrections
    d2, brain, shell, holes, rep = correct(img.data)

    # STEP 2 - smooth brightness offset (additive) and vessels
    d3, field = P.additive_bias_field(d2, brain, float(img.voxel_size[0]))
    bright, bright_thr = P.bright_voxels(d3, brain)
    rep["bright_threshold"] = round(bright_thr, 3)

    # STEP 3 - cerebellum
    cb, brainstem, _, cb_info = cerebellum.isolate(d3, brain)
    rep["cerebellum"] = cb_info                         # midline etc., used by qc.py
    rep["cerebellum_ml"] = round(cb.sum() * ml, 1)

    # STEP 4 - white matter
    d4, sigma = W.denoise(d3, cb)
    wm, frac, _, wm_info = W.segment(d4, cb, bright, brainstem=brainstem, voxel_mm=float(img.voxel_size[0]))
    rep["white_matter"] = wm_info                       # class means etc., used by qc.py
    rep["wm_ml"] = round(wm.sum() * ml, 1)

    save(os.path.join(outdir, "corrected_T1w.nii.gz"), np.where(brain, d3, 0), img)
    save(os.path.join(outdir, "cerebellum_mask.nii.gz"), cb, img, np.uint8)
    save(os.path.join(outdir, "cerebellar_wm_mask.nii.gz"), wm, img, np.uint8)
    save(os.path.join(outdir, "cerebellar_wm_fraction.nii.gz"), frac, img)
    save(os.path.join(outdir, "excluded_voxels.nii.gz"),
         shell.astype(np.uint8) + 2 * holes.astype(np.uint8) + 4 * bright.astype(np.uint8), img, np.uint8)
    np.save(os.path.join(outdir, "_d4_denoised.npy"), d4)        # used by qc.py
    np.save(os.path.join(outdir, "_brainstem_label.npy"), brainstem)
    rep["runtime_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=2, default=float)
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
