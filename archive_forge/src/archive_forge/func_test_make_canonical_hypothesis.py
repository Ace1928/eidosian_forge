import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
@pytest.mark.slow
@hypothesis.given(dtype=hynp.nested_dtypes())
def test_make_canonical_hypothesis(self, dtype):
    canonical = np.result_type(dtype)
    self.check_canonical(dtype, canonical)
    two_arg_result = np.result_type(dtype, dtype)
    assert np.can_cast(two_arg_result, canonical, casting='no')