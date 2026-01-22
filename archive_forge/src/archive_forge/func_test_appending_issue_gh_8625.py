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
def test_appending_issue_gh_8625():
    stream = BytesIO()
    with make_simple(stream, mode='w') as f:
        f.createDimension('x', 2)
        f.createVariable('x', float, ('x',))
        f.variables['x'][...] = 1
        f.flush()
        contents = stream.getvalue()
    stream = BytesIO(contents)
    with netcdf_file(stream, mode='a') as f:
        f.variables['x'][...] = 2