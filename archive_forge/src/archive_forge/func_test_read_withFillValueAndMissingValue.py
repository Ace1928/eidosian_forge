import os
from os.path import join as pjoin, dirname
import shutil
import tempfile
import warnings
from io import BytesIO
from glob import glob
from contextlib import contextmanager
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.io import netcdf_file
from scipy._lib._tmpdirs import in_tempdir
def test_read_withFillValueAndMissingValue():
    IRRELEVANT_VALUE = 9999
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
    with netcdf_file(fname, maskandscale=True) as f:
        vardata = f.variables['var3_fillvalAndMissingValue'][:]
        assert_mask_matches(vardata, [True, False, False])
        assert_equal(vardata, [IRRELEVANT_VALUE, 2, 3])