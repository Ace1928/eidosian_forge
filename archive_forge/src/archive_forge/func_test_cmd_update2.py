import os
import numpy as np
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm.base as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
from nipype.interfaces.spm.base import SPMCommandInputSpec
from nipype.interfaces.base import traits
def test_cmd_update2():

    class TestClass(spm.SPMCommand):
        _jobtype = 'jobtype'
        _jobname = 'jobname'
        input_spec = spm.SPMCommandInputSpec
    dc = TestClass()
    assert dc.jobtype == 'jobtype'
    assert dc.jobname == 'jobname'