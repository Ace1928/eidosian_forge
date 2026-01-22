import os
import numpy as np
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm.base as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
from nipype.interfaces.spm.base import SPMCommandInputSpec
from nipype.interfaces.base import traits
def test_make_matlab_command(create_files_in_directory):

    class TestClass(spm.SPMCommand):
        _jobtype = 'jobtype'
        _jobname = 'jobname'
        input_spec = spm.SPMCommandInputSpec
    dc = TestClass()
    filelist, outdir = create_files_in_directory
    contents = {'contents': [1, 2, 3, 4]}
    script = dc._make_matlab_command([contents])
    assert 'jobs{1}.spm.jobtype.jobname.contents(3) = 3;' in script
    dc.inputs.use_v8struct = False
    script = dc._make_matlab_command([contents])
    assert 'jobs{1}.jobtype{1}.jobname{1}.contents(3) = 3;' in script