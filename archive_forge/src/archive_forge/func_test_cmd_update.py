import os
import numpy as np
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm.base as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
from nipype.interfaces.spm.base import SPMCommandInputSpec
from nipype.interfaces.base import traits
@pytest.mark.skipif(no_spm(), reason='spm is not installed')
def test_cmd_update():

    class TestClass(spm.SPMCommand):
        input_spec = spm.SPMCommandInputSpec
    dc = TestClass()
    dc.inputs.matlab_cmd = 'foo'
    assert dc.mlab._cmd == 'foo'