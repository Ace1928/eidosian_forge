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
@pytest.mark.parametrize('dtype', np.typecodes['AllInteger'])
def test_deprecated_raised(self, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        try:
            np.loadtxt(['10.5'], dtype=dtype)
        except ValueError as e:
            assert isinstance(e.__cause__, DeprecationWarning)