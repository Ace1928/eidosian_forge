import os
import tempfile
import numpy as np
import nibabel as nib
from nibabel.cifti2 import cifti2, cifti2_axes
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
@needs_nibabel_data('nitest-cifti2')
def test_read_conte69_dlabel():
    img = nib.load(os.path.join(test_directory, 'Conte69.parcellations_VGD11b.32k_fs_LR.dlabel.nii'))
    arr = img.get_fdata()
    axes = [img.header.get_axis(dim) for dim in range(2)]
    assert isinstance(axes[0], cifti2_axes.LabelAxis)
    assert len(axes[0]) == 3
    assert (axes[0].name == ['Composite Parcellation-lh (FRB08_OFP03_retinotopic)', 'Brodmann lh (from colin.R via pals_R-to-fs_LR)', 'MEDIAL WALL lh (fs_LR)']).all()
    assert axes[0].label[1][70] == ('19_B05', (1.0, 0.867, 0.467, 1.0))
    assert (axes[0].meta == [{}] * 3).all()
    check_Conte69(axes[1])
    check_rewrite(arr, axes)