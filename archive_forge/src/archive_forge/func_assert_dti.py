import os
import pytest
import shutil
from nipype.interfaces.dcm2nii import Dcm2niix
def assert_dti(res):
    """Some assertions we will make"""
    assert res.outputs.converted_files
    assert res.outputs.bvals
    assert res.outputs.bvecs
    outputs = [y for x, y in res.outputs.get().items()]
    if res.inputs.get('bids_format'):
        assert len(set(map(len, outputs))) == 1
    else:
        assert not res.outputs.bids