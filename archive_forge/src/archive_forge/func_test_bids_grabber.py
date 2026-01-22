import os
import copy
import simplejson
import glob
import os.path as op
from subprocess import Popen
import hashlib
from collections import namedtuple
import pytest
import nipype
import nipype.interfaces.io as nio
from nipype.interfaces.base.traits_extension import isdefined
from nipype.interfaces.base import Undefined, TraitError
from nipype.utils.filemanip import dist_is_editable
from subprocess import check_call, CalledProcessError
@pytest.mark.skipif(not have_pybids, reason='Pybids is not installed')
@pytest.mark.skipif(not dist_is_editable('pybids'), reason='Pybids is not installed in editable mode')
def test_bids_grabber(tmpdir):
    tmpdir.chdir()
    bg = nio.BIDSDataGrabber()
    bg.inputs.base_dir = os.path.join(datadir, 'ds005')
    bg.inputs.subject = '01'
    results = bg.run()
    assert 'sub-01_T1w.nii.gz' in map(os.path.basename, results.outputs.T1w)
    assert 'sub-01_task-mixedgamblestask_run-01_bold.nii.gz' in map(os.path.basename, results.outputs.bold)