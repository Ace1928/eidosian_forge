import tempfile
import shutil
from os import path
from glob import iglob
import re
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import pytest
from scipy.io import (FortranFile,
def test_fortran_bogus_size(tmpdir):
    filename = path.join(str(tmpdir), 'scratch')
    np.random.seed(1)
    with FortranFile(filename, 'w') as f:
        f.write_record(np.random.randn(5))
        f.write_record(np.random.randn(3))
    with open(filename, 'w+b') as f:
        f.write(b'\xff\xff')
    with FortranFile(filename, 'r') as f:
        with pytest.raises(FortranFormattingError):
            f.read_reals()