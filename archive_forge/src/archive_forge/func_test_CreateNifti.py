import pytest
import os
import nibabel as nb
from nipype.algorithms import misc
from nipype.utils.filemanip import fname_presuffix
from nipype.testing.fixtures import create_analyze_pair_file_in_directory
from nipype.testing import example_data
def test_CreateNifti(create_analyze_pair_file_in_directory):
    filelist, outdir = create_analyze_pair_file_in_directory
    create_nifti = misc.CreateNifti()
    with pytest.raises(ValueError):
        create_nifti.run()
    create_nifti.inputs.header_file = filelist[0]
    create_nifti.inputs.data_file = fname_presuffix(filelist[0], '', '.img', use_ext=False)
    result = create_nifti.run()
    assert os.path.exists(result.outputs.nifti_file)
    assert nb.load(result.outputs.nifti_file)