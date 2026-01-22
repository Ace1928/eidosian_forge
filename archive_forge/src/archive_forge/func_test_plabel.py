import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def test_plabel():
    label_map = create_label_map((0,))
    parcel_map = create_parcel_map((1,))
    matrix = ci.Cifti2Matrix()
    matrix.extend((label_map, parcel_map))
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(2, 4)
    img = ci.Cifti2Image(data, hdr)
    with InTemporaryDirectory():
        ci.save(img, 'test.plabel.nii')
        img2 = ci.load('test.plabel.nii')
        assert img.nifti_header.get_intent()[0] == 'ConnUnknown'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        check_label_map(img2.header.matrix.get_index_map(0))
        check_parcel_map(img2.header.matrix.get_index_map(1))
        del img2