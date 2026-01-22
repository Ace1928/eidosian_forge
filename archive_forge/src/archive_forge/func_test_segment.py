import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
@pytest.mark.skipif(no_spm(), reason='spm is not installed')
def test_segment():
    if spm.Info.name() == 'SPM12':
        assert spm.Segment()._jobtype == 'tools'
        assert spm.Segment()._jobname == 'oldseg'
    else:
        assert spm.Segment()._jobtype == 'spatial'
        assert spm.Segment()._jobname == 'preproc'