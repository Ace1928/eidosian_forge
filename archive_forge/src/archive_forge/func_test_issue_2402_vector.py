from dataclasses import dataclass
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.lin_ops.canon_backend import TensorRepresentation
from cvxpy.utilities.coeff_extractor import CoeffExtractor
def test_issue_2402_vector():
    """
    This slight modification with the ridge_coef as a vector also failed
    with a different error due to a dimension mismatch.
    """
    r = np.array([-0.48, 0.11, 0.09, -0.39, 0.03])
    Sigma = np.array([[0.00024, 0.00013, 0.0002, 0.00016, 0.0002], [0.00013, 0.00028, 0.00021, 0.00017, 0.00015], [0.0002, 0.00021, 0.00058, 0.00033, 0.00023], [0.00016, 0.00017, 0.00033, 0.00069, 0.00021], [0.0002, 0.00015, 0.00023, 0.00021, 0.00036]])
    w = cp.Variable(5)
    risk_aversion = cp.Parameter(value=2.0, nonneg=True)
    ridge_coef = cp.Parameter(5, value=np.arange(5), nonneg=True)
    obj_func = r @ w - risk_aversion * cp.quad_form(w, Sigma) - cp.sum(cp.multiply(cp.multiply(ridge_coef, np.array([5, 6, 7, 8, 9])), cp.square(w)))
    objective = cp.Maximize(obj_func)
    fixed_w = np.array([10, 11, 12, 13, 14])
    constraints = [w == fixed_w]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    expected_value = r @ fixed_w - risk_aversion.value * np.dot(fixed_w, np.dot(Sigma, fixed_w)) - np.sum(ridge_coef.value * np.array([5, 6, 7, 8, 9]) * np.square(fixed_w))
    assert np.isclose(prob.value, expected_value)
    assert np.allclose(w.value, fixed_w)