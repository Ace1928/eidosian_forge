import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
from nipype.interfaces import freesurfer
from nipype.interfaces.freesurfer import Info
from nipype import LooseVersion
def test_FSVersion():
    """Check that FSVersion is a string that can be compared with LooseVersion"""
    assert isinstance(freesurfer.preprocess.FSVersion, str)
    assert LooseVersion(freesurfer.preprocess.FSVersion) >= LooseVersion('0')