"""NIfTI input/output with nibabel (the standard Python NIfTI library)."""
import nibabel as nib
import numpy as np


class Image:
    """Small container: voxel data (float32), affine (4x4) and the original header."""

    def __init__(self, data, affine, header):
        self.data = data
        self.affine = affine
        self.header = header

    @property
    def voxel_size(self):
        return np.sqrt((self.affine[:3, :3] ** 2).sum(axis=0))


def load(path):
    img = nib.load(path)
    return Image(np.asanyarray(img.dataobj).astype(np.float32), img.affine, img.header)


def save(path, data, ref, dtype=np.float32):
    """Save `data` with the geometry and header of `ref` (an Image)."""
    hdr = ref.header.copy()
    hdr.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(np.asarray(data).astype(dtype), ref.affine, hdr), path)
