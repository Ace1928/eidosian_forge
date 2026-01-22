import os
from copy import deepcopy
import pytest
import pdb
from nipype.utils.filemanip import split_filename, ensure_list
from .. import preprocess as fsl
from nipype.interfaces.fsl import Info
from nipype.interfaces.base import File, TraitError, Undefined, isdefined
from nipype.interfaces.fsl import no_fsl
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_first_genfname():
    first = fsl.FIRST()
    first.inputs.out_file = 'segment.nii'
    first.inputs.output_type = 'NIFTI_GZ'
    value = first._gen_fname(basename='original_segmentations')
    expected_value = os.path.abspath('segment_all_fast_origsegs.nii.gz')
    assert value == expected_value
    first.inputs.method = 'none'
    value = first._gen_fname(basename='original_segmentations')
    expected_value = os.path.abspath('segment_all_none_origsegs.nii.gz')
    assert value == expected_value
    first.inputs.method = 'auto'
    first.inputs.list_of_specific_structures = ['L_Hipp', 'R_Hipp']
    value = first._gen_fname(basename='original_segmentations')
    expected_value = os.path.abspath('segment_all_none_origsegs.nii.gz')
    assert value == expected_value