import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_all_expressions(self) -> None:
    complex_X = Constant(np.array([[1.0, 4.0, 7.0], [2.0, -4.0 + 3j, 3.0], [99.0, -2.0 - 9j, 2.4]]))
    X = Constant(np.array([[1.0, 4.0, 7.0], [2.0, -4.0, 3.0], [99.0, -2.0, 2.4]]))
    for method in ['conj']:
        fn = getattr(cp, method)
        method_fn = getattr(complex_X, method)
        assert fn(complex_X).shape == method_fn().shape
        assert np.allclose(fn(complex_X).value, method_fn().value)
    for method in ['conj', 'trace', 'cumsum', 'max', 'min', 'mean', 'ptp', 'prod', 'sum', 'std', 'var']:
        fn = getattr(cp, method)
        method_fn = getattr(X, method)
        assert fn(X).shape == method_fn().shape
        assert np.allclose(fn(X).value, method_fn().value)
    for method in ['cumsum']:
        for axis in [None, 0, 1]:
            fn = getattr(cp, method)(X, axis)
            method_fn = getattr(X, method)(axis)
            assert fn.shape == method_fn.shape
            assert np.allclose(fn.value, method_fn.value)
    for method in ['max', 'mean', 'min', 'prod', 'ptp', 'sum']:
        for axis in [None, 0, 1]:
            for keepdims in [True, False]:
                fn = getattr(cp, method)(X, axis, keepdims)
                method_fn = getattr(X, method)(axis, keepdims=keepdims)
                assert fn.shape == method_fn.shape
                assert np.allclose(fn.value, method_fn.value)
    for method in ['std']:
        for axis in [None, 0, 1]:
            for keepdims in [True, False]:
                for ddof in [0, 1, 2]:
                    fn = getattr(cp, method)(X, axis, keepdims, ddof=ddof)
                    method_fn = getattr(X, method)(axis, keepdims=keepdims, ddof=ddof)
                    assert fn.shape == method_fn.shape
                    assert np.allclose(fn.value, method_fn.value)
    for method in ['var']:
        for ddof in [0, 1, 2]:
            fn = getattr(cp, method)(X, ddof=ddof)
            method_fn = getattr(X, method)(ddof=ddof)
            assert fn.shape == method_fn.shape
            assert np.allclose(fn.value, method_fn.value)