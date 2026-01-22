from itertools import product
from numpy.testing import (assert_, assert_allclose, assert_array_less,
import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._numdiff import group_columns
from scipy.integrate import solve_ivp, RK23, RK45, DOP853, Radau, BDF, LSODA
from scipy.integrate import OdeSolution
from scipy.integrate._ivp.common import num_jac
from scipy.integrate._ivp.base import ConstantDenseOutput
from scipy.sparse import coo_matrix, csc_matrix
def medazko_sparsity(n):
    cols = []
    rows = []
    i = np.arange(n) * 2
    cols.append(i[1:])
    rows.append(i[1:] - 2)
    cols.append(i)
    rows.append(i)
    cols.append(i)
    rows.append(i + 1)
    cols.append(i[:-1])
    rows.append(i[:-1] + 2)
    i = np.arange(n) * 2 + 1
    cols.append(i)
    rows.append(i)
    cols.append(i)
    rows.append(i - 1)
    cols = np.hstack(cols)
    rows = np.hstack(rows)
    return coo_matrix((np.ones_like(cols), (cols, rows)))