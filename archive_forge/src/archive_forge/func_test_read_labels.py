import io
from os.path import dirname
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from packaging.version import Version
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.cifti2.parse_cifti2 import _Cifti2AsNiftiHeader
from nibabel.tests import test_nifti2 as tn2
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from nibabel.tmpdirs import InTemporaryDirectory
@needs_nibabel_data('nitest-cifti2')
def test_read_labels():
    img = ci.Cifti2Image.from_filename(DATA_FILE5)
    label_mapping = img.header.matrix.get_index_map(0)
    expected_names = ['Composite Parcellation-lh (FRB08_OFP03_retinotopic)', 'Brodmann lh (from colin.R via pals_R-to-fs_LR)', 'MEDIAL WALL lh (fs_LR)']
    assert img.shape[0] == len(expected_names)
    assert len(list(label_mapping.named_maps)) == len(expected_names)
    some_expected_labels = {0: ('???', (0.667, 0.667, 0.667, 0.0)), 1: ('MEDIAL.WALL', (0.075, 0.075, 0.075, 1.0)), 2: ('BA2_FRB08', (0.467, 0.459, 0.055, 1.0)), 3: ('BA1_FRB08', (0.475, 0.722, 0.859, 1.0)), 4: ('BA3b_FRB08', (0.855, 0.902, 0.286, 1.0)), 5: ('BA4p_FRB08', (0.902, 0.573, 0.122, 1.0)), 89: ('36_B05', (0.467, 0.0, 0.129, 1.0)), 90: ('35_B05', (0.467, 0.067, 0.067, 1.0)), 91: ('28_B05', (0.467, 0.337, 0.271, 1.0)), 92: ('29_B05', (0.267, 0.0, 0.529, 1.0)), 93: ('26_B05', (0.757, 0.2, 0.227, 1.0)), 94: ('33_B05', (0.239, 0.082, 0.373, 1.0)), 95: ('13b_OFP03', (1.0, 1.0, 0.0, 1.0))}
    for named_map, name in zip(label_mapping.named_maps, expected_names):
        assert named_map.map_name == name
        assert len(named_map.metadata) == 0
        assert len(named_map.label_table) == 96
        for index, (label, rgba) in some_expected_labels.items():
            assert named_map.label_table[index].label == label
            assert named_map.label_table[index].rgba == rgba