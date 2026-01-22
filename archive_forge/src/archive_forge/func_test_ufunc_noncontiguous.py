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
@np._no_nep50_warning()
def test_ufunc_noncontiguous(ufunc):
    """
    Check that contiguous and non-contiguous calls to ufuncs
    have the same results for values in range(9)
    """
    for typ in ufunc.types:
        if any(set('O?mM') & set(typ)):
            continue
        inp, out = typ.split('->')
        args_c = [np.empty(6, t) for t in inp]
        args_n = [np.empty(18, t)[::3] for t in inp]
        for a in args_c:
            a.flat = range(1, 7)
        for a in args_n:
            a.flat = range(1, 7)
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings('always')
            res_c = ufunc(*args_c)
            res_n = ufunc(*args_n)
        if len(out) == 1:
            res_c = (res_c,)
            res_n = (res_n,)
        for c_ar, n_ar in zip(res_c, res_n):
            dt = c_ar.dtype
            if np.issubdtype(dt, np.floating):
                res_eps = np.finfo(dt).eps
                tol = 2 * res_eps
                assert_allclose(res_c, res_n, atol=tol, rtol=tol)
            else:
                assert_equal(c_ar, n_ar)