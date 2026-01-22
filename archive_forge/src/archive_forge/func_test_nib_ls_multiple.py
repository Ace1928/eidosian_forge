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
@unittest.skipUnless(load_small_file(), "Can't load the small.mnc file")
@script_test
def test_nib_ls_multiple():
    fnames = [pjoin(DATA_PATH, f) for f in ('example4d.nii.gz', 'example_nifti2.nii.gz', 'small.mnc', 'nifti2.hdr')]
    code, stdout, stderr = run_command(['nib-ls'] + fnames)
    stdout_lines = stdout.split('\n')
    assert len(stdout_lines) == 4
    ln = max((len(f) for f in fnames))
    i_str = ' i' if sys.byteorder == 'little' else ' <i'
    assert [l[ln:ln + len(i_str)] for l in stdout_lines] == [i_str] * 4, f"Type sub-string didn't start with '{i_str}'. Full output was: {stdout_lines}"
    assert [l[l.index('['):] for l in stdout_lines] == ['[128,  96,  24,   2] 2.00x2.00x2.20x2000.00  #exts: 2 sform', '[ 32,  20,  12,   2] 2.00x2.00x2.20x2000.00  #exts: 2 sform', '[ 18,  28,  29]      9.00x8.00x7.00', '[ 91, 109,  91]      2.00x2.00x2.00']
    code, stdout, stderr = run_command(['nib-ls', '-s'] + fnames)
    stdout_lines = stdout.split('\n')
    assert len(stdout_lines) == 4
    assert [l[l.index('['):] for l in stdout_lines] == ['[128,  96,  24,   2] 2.00x2.00x2.20x2000.00  #exts: 2 sform [229725] [2, 1.2e+03]', '[ 32,  20,  12,   2] 2.00x2.00x2.20x2000.00  #exts: 2 sform [15360]  [46, 7.6e+02]', '[ 18,  28,  29]      9.00x8.00x7.00                         [14616]  [0.12, 93]', '[ 91, 109,  91]      2.00x2.00x2.00                          !error']