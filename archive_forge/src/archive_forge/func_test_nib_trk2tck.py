import csv
import os
import shutil
import sys
import unittest
from glob import glob
from os.path import abspath, basename, dirname, exists
from os.path import join as pjoin
from os.path import splitext
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
import nibabel as nib
from ..loadsave import load
from ..orientations import aff2axcodes, inv_ornt_aff
from ..testing import assert_data_similar, assert_dt_equal, assert_re_in
from ..tmpdirs import InTemporaryDirectory
from .nibabel_data import needs_nibabel_data
from .scriptrunner import ScriptRunner
from .test_parrec import DTI_PAR_BVALS, DTI_PAR_BVECS
from .test_parrec import EXAMPLE_IMAGES as PARREC_EXAMPLES
from .test_parrec_data import AFF_OFF, BALLS
@script_test
def test_nib_trk2tck():
    simple_trk = pjoin(DATA_PATH, 'simple.trk')
    standard_trk = pjoin(DATA_PATH, 'standard.trk')
    with InTemporaryDirectory() as tmpdir:
        shutil.copy(simple_trk, tmpdir)
        shutil.copy(standard_trk, tmpdir)
        simple_trk = pjoin(tmpdir, 'simple.trk')
        standard_trk = pjoin(tmpdir, 'standard.trk')
        simple_tck = pjoin(tmpdir, 'simple.tck')
        standard_tck = pjoin(tmpdir, 'standard.tck')
        cmd = ['nib-trk2tck', simple_trk]
        code, stdout, stderr = run_command(cmd)
        assert len(stdout) == 0
        assert os.path.isfile(simple_tck)
        trk = nib.streamlines.load(simple_trk)
        tck = nib.streamlines.load(simple_tck)
        assert (tck.streamlines.get_data() == trk.streamlines.get_data()).all()
        assert isinstance(tck, nib.streamlines.TckFile)
        cmd = ['nib-trk2tck', simple_tck]
        code, stdout, stderr = run_command(cmd)
        assert 'Skipping non TRK file' in stdout
        cmd = ['nib-trk2tck', simple_trk]
        code, stdout, stderr = run_command(cmd)
        assert 'Skipping existing file' in stdout
        cmd = ['nib-trk2tck', '--force', simple_trk, standard_trk]
        code, stdout, stderr = run_command(cmd)
        assert len(stdout) == 0
        trk = nib.streamlines.load(standard_trk)
        tck = nib.streamlines.load(standard_tck)
        assert (tck.streamlines.get_data() == trk.streamlines.get_data()).all()