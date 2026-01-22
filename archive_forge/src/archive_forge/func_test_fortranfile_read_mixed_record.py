import tempfile
import shutil
from os import path
from glob import iglob
import re
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import pytest
from scipy.io import (FortranFile,
def test_fortranfile_read_mixed_record():
    filename = path.join(DATA_PATH, 'fortran-3x3d-2i.dat')
    with FortranFile(filename, 'r', '<u4') as f:
        record = f.read_record('(3,3)<f8', '2<i4')
    ax = np.arange(3 * 3).reshape(3, 3).astype(np.float64)
    bx = np.array([-1, -2], dtype=np.int32)
    assert_equal(record[0], ax.T)
    assert_equal(record[1], bx.T)