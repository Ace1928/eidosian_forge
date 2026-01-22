import tempfile
import shutil
from os import path
from glob import iglob
import re
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import pytest
from scipy.io import (FortranFile,
def test_fortranfiles_read():
    for filename in iglob(path.join(DATA_PATH, 'fortran-*-*x*x*.dat')):
        m = re.search('fortran-([^-]+)-(\\d+)x(\\d+)x(\\d+).dat', filename, re.I)
        if not m:
            raise RuntimeError("Couldn't match %s filename to regex" % filename)
        dims = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
        dtype = m.group(1).replace('s', '<')
        f = FortranFile(filename, 'r', '<u4')
        data = f.read_record(dtype=dtype).reshape(dims, order='F')
        f.close()
        expected = np.arange(np.prod(dims)).reshape(dims).astype(dtype)
        assert_equal(data, expected)