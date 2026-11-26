"""STEP 5 (optional cross-check) - atlas-based cerebellum + deep-nuclei labels via
ANTs registration (antspyx).

Idea: warp a standard brain (MNI152NLin6Asym) onto this
subject's brain with ANTs (rigid -> affine -> SyN nonlinear), then carry the
Diedrichsen et al. (2009/2011) cerebellar atlas (lobules + dentate /
interposed / fastigial nuclei) along the same transform into subject space.

Usage:
  python step5_atlas_registration.py <subject_T1w.nii.gz> <atlas_dir> <outdir>
atlas_dir must contain (all public downloads, see README):
  tpl-MNI152NLin6Asym_res-01_T1w.nii.gz
  tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii.gz
  atl-Anatom_space-MNI_dseg.nii          (Diedrichsen 2009, MNI152NLin6AsymC)
"""
import os, sys, time, json
import numpy as np
import ants

subj_f, adir, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
t0 = time.time()

subj = ants.image_read(subj_f)
# registration is done at 1 mm (speed); the transform is in mm, so it is
# applied afterwards on the full 0.55 mm grid.
subj1 = ants.resample_image(subj, (1, 1, 1), use_voxels=False, interp_type=0)

tpl = ants.image_read(os.path.join(adir, "tpl-MNI152NLin6Asym_res-01_T1w.nii.gz"))
tmask = ants.image_read(os.path.join(adir, "tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii.gz"))
tpl_brain = tpl * tmask            # subject is skull-stripped -> strip template too

# Stage 1: whole-brain rigid + affine + SyN (template -> subject)
reg = ants.registration(fixed=subj1, moving=tpl_brain, type_of_transform="SyN",
                        aff_metric="mattes", syn_metric="mattes", syn_sampling=32,
                        reg_iterations=(100, 70, 40), verbose=False)
print(f"registration done in {time.time()-t0:.0f} s")
ants.image_write(reg["warpedmovout"], os.path.join(out, "template_in_subject_1mm.nii.gz"))

# carry the atlas into the subject at full resolution
atl = ants.image_read(os.path.join(adir, "atl-Anatom_space-MNI_dseg.nii"))
lab = ants.apply_transforms(fixed=subj, moving=atl, transformlist=reg["fwdtransforms"],
                            interpolator="genericLabel")
ants.image_write(lab, os.path.join(out, "atlas_labels_in_subject.nii.gz"))

L = lab.numpy()
vox = float(np.prod(subj.spacing)) / 1000.0  # mL per voxel
cereb = (L >= 1) & (L <= 34)
dent = (L == 29) | (L == 30)
nuclei = (L >= 29) & (L <= 34)
for name, m in [("atlas_cerebellum_mask", cereb), ("atlas_dentate_mask", dent),
                ("atlas_nuclei_mask", nuclei)]:
    ants.image_write(subj.new_image_like(m.astype("float32")),
                     os.path.join(out, name + ".nii.gz"))
res = {"atlas_cerebellum_mL": cereb.sum() * vox, "atlas_dentate_mL": dent.sum() * vox,
       "atlas_nuclei_mL": nuclei.sum() * vox, "runtime_s": time.time() - t0}
json.dump(res, open(os.path.join(out, "atlas_report.json"), "w"), indent=2)
print(res)
