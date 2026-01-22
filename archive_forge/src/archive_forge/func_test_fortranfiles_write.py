import tempfile
import shutil
from os import path
from glob import iglob
import re
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import pytest
from scipy.io import (FortranFile,
def test_fortranfiles_write():
    for filename in iglob(path.join(DATA_PATH, 'fortran-*-*x*x*.dat')):
        m = re.search('fortran-([^-]+)-(\\d+)x(\\d+)x(\\d+).dat', filename, re.I)
        if not m:
            raise RuntimeError("Couldn't match %s filename to regex" % filename)
        dims = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
        dtype = m.group(1).replace('s', '<')
        data = np.arange(np.prod(dims)).reshape(dims).astype(dtype)
        tmpdir = tempfile.mkdtemp()
        try:
            testFile = path.join(tmpdir, path.basename(filename))
            f = FortranFile(testFile, 'w', '<u4')
            f.write_record(data.T)
            f.close()
            originalfile = open(filename, 'rb')
            newfile = open(testFile, 'rb')
            assert_equal(originalfile.read(), newfile.read(), err_msg=filename)
            originalfile.close()
            newfile.close()
        finally:
            shutil.rmtree(tmpdir)