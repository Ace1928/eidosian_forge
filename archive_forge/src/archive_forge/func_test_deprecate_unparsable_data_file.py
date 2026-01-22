import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
@pytest.mark.parametrize('invalid_str', [',invalid_data', 'invalid_sep'])
def test_deprecate_unparsable_data_file(self, invalid_str):
    x = np.array([1.51, 2, 3.51, 4], dtype=float)
    with tempfile.TemporaryFile(mode='w') as f:
        x.tofile(f, sep=',', format='%.2f')
        f.write(invalid_str)
        f.seek(0)
        self.assert_deprecated(lambda: np.fromfile(f, sep=','))
        f.seek(0)
        self.assert_deprecated(lambda: np.fromfile(f, sep=',', count=5))
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            f.seek(0)
            res = np.fromfile(f, sep=',', count=4)
            assert_array_equal(res, x)