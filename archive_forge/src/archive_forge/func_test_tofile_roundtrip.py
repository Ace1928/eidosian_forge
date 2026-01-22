import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
@pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
def test_tofile_roundtrip(self):
    with temppath() as path:
        self.tgt.tofile(path, sep=' ')
        res = np.fromfile(path, dtype=np.longdouble, sep=' ')
    assert_equal(res, self.tgt)