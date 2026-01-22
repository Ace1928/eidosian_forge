import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
@pytest.mark.slow
@given(sampled_from(reasonable_operators_for_scalars), hynp.scalar_dtypes(), hynp.scalar_dtypes())
def test_array_scalar_ufunc_dtypes(op, dt1, dt2):
    arr1 = np.array(2, dtype=dt1)
    arr2 = np.array(3, dtype=dt2)
    check_ufunc_scalar_equivalence(op, arr1, arr2)