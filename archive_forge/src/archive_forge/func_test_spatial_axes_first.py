import warnings
from os.path import dirname
from os.path import join as pjoin
import numpy as np
import pytest
import nibabel as nib
from nibabel import imageclasses
from nibabel.analyze import AnalyzeImage
from nibabel.imageclasses import spatial_axes_first
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from nibabel.optpkg import optional_package
def test_spatial_axes_first():
    affine = np.eye(4)
    for shape in ((2, 3), (4, 3, 2), (5, 4, 1, 2), (2, 3, 5, 2, 1)):
        for img_class in (AnalyzeImage, Nifti1Image, Nifti2Image):
            data = np.zeros(shape)
            img = img_class(data, affine)
            assert spatial_axes_first(img)
    for fname in MINC_3DS:
        img = nib.load(pjoin(DATA_DIR, fname))
        assert len(img.shape) == 3
        assert spatial_axes_first(img)
    for fname in MINC_4DS:
        img = nib.load(pjoin(DATA_DIR, fname))
        assert len(img.shape) == 4
        assert not spatial_axes_first(img)