import itertools
import platform
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy import optimize
from scipy.optimize._minimize import Bounds, NonlinearConstraint
from scipy.optimize._minimize import (MINIMIZE_METHODS,
from scipy.optimize._linprog import LINPROG_METHODS
from scipy.optimize._root import ROOT_METHODS
from scipy.optimize._root_scalar import ROOT_SCALAR_METHODS
from scipy.optimize._qap import QUADRATIC_ASSIGNMENT_METHODS
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
from scipy.optimize._optimize import MemoizeJac, show_options, OptimizeResult
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.sparse import (coo_matrix, csc_matrix, csr_matrix, coo_array,
def test_show_options():
    solver_methods = {'minimize': MINIMIZE_METHODS, 'minimize_scalar': MINIMIZE_SCALAR_METHODS, 'root': ROOT_METHODS, 'root_scalar': ROOT_SCALAR_METHODS, 'linprog': LINPROG_METHODS, 'quadratic_assignment': QUADRATIC_ASSIGNMENT_METHODS}
    for solver, methods in solver_methods.items():
        for method in methods:
            show_options(solver, method)
    unknown_solver_method = {'minimize': 'ekki', 'maximize': 'cg', 'maximize_scalar': 'ekki'}
    for solver, method in unknown_solver_method.items():
        assert_raises(ValueError, show_options, solver, method)