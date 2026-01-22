import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
def test_coregister():
    assert spm.Coregister._jobtype == 'spatial'
    assert spm.Coregister._jobname == 'coreg'
    assert spm.Coregister().inputs.jobtype == 'estwrite'