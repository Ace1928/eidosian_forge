import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('bad_col_ind', (3, -3))
def test_converters_dict_raises_non_col_key(bad_col_ind):
    data = StringIO('1 2\n3 4')
    with pytest.raises(ValueError, match='converter specified for column'):
        np.loadtxt(data, converters={bad_col_ind: int})