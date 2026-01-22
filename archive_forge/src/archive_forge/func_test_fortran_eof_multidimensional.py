import tempfile
import shutil
from os import path
from glob import iglob
import re
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import pytest
from scipy.io import (FortranFile,
def test_fortran_eof_multidimensional(tmpdir):
    filename = path.join(str(tmpdir), 'scratch')
    n, m, q = (3, 5, 7)
    dt = np.dtype([('field', np.float64, (n, m))])
    a = np.zeros(q, dtype=dt)
    with FortranFile(filename, 'w') as f:
        f.write_record(a[0])
        f.write_record(a)
        f.write_record(a)
    with open(filename, 'ab') as f:
        f.truncate(path.getsize(filename) - 20)
    with FortranFile(filename, 'r') as f:
        assert len(f.read_record(dtype=dt)) == 1
        assert len(f.read_record(dtype=dt)) == q
        with pytest.raises(FortranFormattingError):
            f.read_record(dtype=dt)