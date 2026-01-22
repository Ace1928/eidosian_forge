from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.xfail(IS_PYPY, reason='`int(np.complex128(3))` fails on PyPy')
@pytest.mark.filterwarnings('ignore::numpy.ComplexWarning')
@pytest.mark.parametrize('cast_to', scalar_instances())
def test_scalar_coercion_same_as_cast_and_assignment(self, cast_to):
    """
        Test that in most cases:
           * `np.array(scalar, dtype=dtype)`
           * `np.empty((), dtype=dtype)[()] = scalar`
           * `np.array(scalar).astype(dtype)`
        should behave the same.  The only exceptions are parametric dtypes
        (mainly datetime/timedelta without unit) and void without fields.
        """
    dtype = cast_to.dtype
    for scalar in scalar_instances(times=False):
        scalar = scalar.values[0]
        if dtype.type == np.void:
            if scalar.dtype.fields is not None and dtype.fields is None:
                with pytest.raises(TypeError):
                    np.array(scalar).astype(dtype)
                np.array(scalar, dtype=dtype)
                np.array([scalar], dtype=dtype)
                continue
        try:
            cast = np.array(scalar).astype(dtype)
        except (TypeError, ValueError, RuntimeError):
            with pytest.raises(Exception):
                np.array(scalar, dtype=dtype)
            if isinstance(scalar, rational) and np.issubdtype(dtype, np.signedinteger):
                return
            with pytest.raises(Exception):
                np.array([scalar], dtype=dtype)
            res = np.zeros((), dtype=dtype)
            with pytest.raises(Exception):
                res[()] = scalar
            return
        arr = np.array(scalar, dtype=dtype)
        assert_array_equal(arr, cast)
        ass = np.zeros((), dtype=dtype)
        ass[()] = scalar
        assert_array_equal(ass, cast)