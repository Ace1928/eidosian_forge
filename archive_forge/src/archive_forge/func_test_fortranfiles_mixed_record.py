import tempfile
import shutil
from os import path
from glob import iglob
import re
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import pytest
from scipy.io import (FortranFile,
def test_fortranfiles_mixed_record():
    filename = path.join(DATA_PATH, 'fortran-mixed.dat')
    with FortranFile(filename, 'r', '<u4') as f:
        record = f.read_record('<i4,<f4,<i8,(2)<f8')
    assert_equal(record['f0'][0], 1)
    assert_allclose(record['f1'][0], 2.3)
    assert_equal(record['f2'][0], 4)
    assert_allclose(record['f3'][0], [5.6, 7.8])