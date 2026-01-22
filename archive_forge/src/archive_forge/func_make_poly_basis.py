from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def make_poly_basis(x, degree, intercept=True):
    """
    given a vector x returns poly=(1, x, x^2, ..., x^degree)
    and its first and second derivative
    """
    if intercept:
        start = 0
    else:
        start = 1
    nobs = len(x)
    basis = np.zeros(shape=(nobs, degree + 1 - start))
    der_basis = np.zeros(shape=(nobs, degree + 1 - start))
    der2_basis = np.zeros(shape=(nobs, degree + 1 - start))
    for i in range(start, degree + 1):
        basis[:, i - start] = x ** i
        der_basis[:, i - start] = i * x ** (i - 1)
        der2_basis[:, i - start] = i * (i - 1) * x ** (i - 2)
    return (basis, der_basis, der2_basis)