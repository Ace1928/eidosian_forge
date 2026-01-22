import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
@pytest.mark.parametrize('ufunc', [getattr(np, x) for x in dir(np) if isinstance(getattr(np, x), np.ufunc)])
def test_ufunc_types(ufunc):
    """
    Check all ufuncs that the correct type is returned. Avoid
    object and boolean types since many operations are not defined for
    for them.

    Choose the shape so even dot and matmul will succeed
    """
    for typ in ufunc.types:
        if 'O' in typ or '?' in typ:
            continue
        inp, out = typ.split('->')
        args = [np.ones((3, 3), t) for t in inp]
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings('always')
            res = ufunc(*args)
        if isinstance(res, tuple):
            outs = tuple(out)
            assert len(res) == len(outs)
            for r, t in zip(res, outs):
                assert r.dtype == np.dtype(t)
        else:
            assert res.dtype == np.dtype(out)