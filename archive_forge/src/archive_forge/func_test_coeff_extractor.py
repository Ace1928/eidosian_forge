from dataclasses import dataclass
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.lin_ops.canon_backend import TensorRepresentation
from cvxpy.utilities.coeff_extractor import CoeffExtractor
def test_coeff_extractor(coeff_extractor):
    """
    This is a unit test for the same problem.
    The variable and parameter namings are derived from the problem above.
    """
    x1 = cp.Variable(2, var_id=1)
    x14 = cp.Variable((1, 1), var_id=14)
    x16 = cp.Variable(var_id=16)
    p2 = cp.Parameter(value=1.0, nonneg=True, id=2)
    p3 = cp.Parameter(value=0.0, nonneg=True, id=3)
    affine_expr = p2 * x14 + p3 * x16
    quad_forms = {x14.id: (p2 * x14, 1, SymbolicQuadForm(x1, cp.Constant(np.eye(2)), cp.quad_form(x1, np.eye(2)))), x16.id: (p3 * x16, 1, SymbolicQuadForm(x1, cp.Constant(np.eye(2)), cp.quad_over_lin(x1, 1.0)))}
    coeffs, constant = coeff_extractor.extract_quadratic_coeffs(affine_expr, quad_forms)
    assert len(coeffs) == 1
    assert np.allclose(coeffs[1]['q'].toarray(), np.zeros((2, 3)))
    P = coeffs[1]['P']
    assert isinstance(P, TensorRepresentation)
    assert np.allclose(P.data, np.ones(4))
    assert np.allclose(P.row, np.array([0, 1, 0, 1]))
    assert np.allclose(P.col, np.array([0, 1, 0, 1]))
    assert P.shape == (2, 2)
    assert np.allclose(P.parameter_offset, np.array([0, 0, 1, 1]))
    assert np.allclose(constant.toarray(), np.zeros(3))