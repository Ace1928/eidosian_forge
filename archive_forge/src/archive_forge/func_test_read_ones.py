import os
import tempfile
import numpy as np
import nibabel as nib
from nibabel.cifti2 import cifti2, cifti2_axes
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
@needs_nibabel_data('nitest-cifti2')
def test_read_ones():
    img = nib.load(os.path.join(test_directory, 'ones.dscalar.nii'))
    arr = img.get_fdata()
    axes = [img.header.get_axis(dim) for dim in range(2)]
    assert (arr == 1).all()
    assert isinstance(axes[0], cifti2_axes.ScalarAxis)
    assert len(axes[0]) == 1
    assert axes[0].name[0] == 'ones'
    assert axes[0].meta[0] == {}
    check_hcp_grayordinates(axes[1])
    img = check_rewrite(arr, axes)
    check_hcp_grayordinates(img.header.get_axis(1))