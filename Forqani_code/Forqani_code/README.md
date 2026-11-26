# Cerebellar white-matter segmentation (0.55 mm MPRAGE-like MP2RAGE)

The pipeline is organised in numbered steps; steps 5 and 6 are an optional atlas cross-check.

```
python run_all_v3.py <input.nii.gz> <outdir>                                 # steps 1-4, about 1-2 min
python qc.py <input.nii.gz> <outdir>                                         # threshold-by-depth test and figures
python step5_atlas_registration.py <outdir>/corrected_T1w.nii.gz <atlas_dir> <atlas_out>   # optional, about 7 min
python step6_atlas_check.py <outdir>/corrected_T1w.nii.gz <outdir> <atlas_out>             # optional
```
Step 1 can also be run on its own: `python step1_image_corrections.py <input.nii.gz> <outdir>`.

In the two-page report, "Step 1" covers code steps 1–2, "Step 2" is code step 3 and "Step 3" is code step 4.

Requirements: numpy, scipy, scikit-image, scikit-learn, matplotlib, nibabel (`pip install -r requirements.txt`; tested versions are listed there). Steps 5 and 6 also need `antspyx`.

## Steps
| Step | File | What it does | sub-405 |
|---|---|---|---|
| 1 | `step1_image_corrections.py` | **Constant-value voxels:** 4.9% of the voxels share one value. Those connected to the mask edge are removed, and interior ones are filled with the median of their neighbours. **Two abnormal sagittal planes** (i = 165–166, compressed contrast) are replaced by interpolation. | 357k removed, 92k filled |
| 2 | `preprocess.py` | A smooth regional brightness offset is removed (additive model, σ = 15 mm). Very bright voxels (vessels, above the WM mean + 4 SD) are flagged. | |
| 3 | `cerebellum.py` | The cerebellum is isolated without an atlas: a brainstem ellipse model plus a marker watershed on dark lines (tentorium). | 136.1 mL |
| 4 | `whitematter.py` | Non-local-means denoising; GM/WM mixture gives a WM fraction f; a sheet filter picks up thin laminae; hysteresis thresholds (0.5 and 0.3); no WM within 1 mm of the surface. | 37.6 mL |
| QC | `qc.py` | Tests the prediction that raising the threshold removes thin superficial WM first; produces figures. | |
| 5 | `step5_atlas_registration.py` | ANTs registration (rigid → affine → SyN) of MNI152NLin6Asym to the subject. The Diedrichsen (2009) cerebellar atlas, with its lobules and deep nuclei, is warped into subject space. | about 7 min |
| 6 | `step6_atlas_check.py` | Compares the atlas and intensity-based cerebellum, locates the dentate nucleus inside the WM mask, and makes a QC figure (attached separately as Supplementary Figure S1, `Forqani_supplementary_figure_S1.pdf`). | Dice 0.89; dentate 3.8 mL, 96% inside WM; WM without dentate 34.0 mL |

Atlas files for step 5 (public downloads):
- `tpl-MNI152NLin6Asym_res-01_T1w.nii.gz` and `tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii.gz` from TemplateFlow
- `atl-Anatom_space-MNI_dseg.nii` from github.com/DiedrichsenLab/cerebellar_atlases

Parameters: the voxel-index parameters in `cerebellum.py` (slice ranges and crop box) were set for sub-405 and would need adapting, or an atlas-based starting mask, for other subjects.

The lobular atlas does not label the central WM and peduncles. It is therefore used only as a cross-check and to locate the nuclei, not as the cerebellar mask.

## Attribution
The order of steps follows the ideas of the Spinoza/NIN cerebellar pipeline (<https://github.com/piloubazin/cerebellar-cortex>, Apache-2.0): denoising → cerebellar masking → tissue clustering → ridge filtering of thin WM branches → dura handling. **No code was copied.** All steps were re-implemented for this contrast.
