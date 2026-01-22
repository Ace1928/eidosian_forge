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
def test_parrec2nii():
    cmd = ['parrec2nii', '--help']
    code, stdout, stderr = run_command(cmd)
    assert stdout.startswith('Usage')
    with InTemporaryDirectory():
        for eg_dict in PARREC_EXAMPLES:
            fname = eg_dict['fname']
            run_command(['parrec2nii', fname])
            out_froot = splitext(basename(fname))[0] + '.nii'
            img = load(out_froot)
            assert img.shape == eg_dict['shape']
            assert_dt_equal(img.get_data_dtype(), eg_dict['dtype'])
            data = img.get_fdata()
            assert_data_similar(data, eg_dict)
            assert_almost_equal(img.header.get_zooms(), eg_dict['zooms'])
            assert len(img.header.extensions) == 0
            del img, data
            code, stdout, stderr = run_command(['parrec2nii', fname], check_code=False)
            assert code == 1
            pr_img = load(fname)
            flipped_data = np.flip(pr_img.get_fdata(), 1)
            base_cmd = ['parrec2nii', '--overwrite', fname]
            check_conversion(base_cmd, flipped_data, out_froot)
            check_conversion(base_cmd + ['--scaling=dv'], flipped_data, out_froot)
            pr_img = load(fname, scaling='fp')
            flipped_data = np.flip(pr_img.get_fdata(), 1)
            check_conversion(base_cmd + ['--scaling=fp'], flipped_data, out_froot)
            unscaled_flipped = np.flip(pr_img.dataobj.get_unscaled(), 1)
            check_conversion(base_cmd + ['--scaling=off'], unscaled_flipped, out_froot)
            run_command(base_cmd + ['--store-header'])
            img = load(out_froot)
            assert len(img.header.extensions) == 1
            del img