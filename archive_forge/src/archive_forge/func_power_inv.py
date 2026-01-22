import numpy as np
from cvxpy import atoms
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import DivExpression
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.affine.unary_operators import NegExpression
def power_inv(t):
    if expr.p.value == 1:
        return t
    return atoms.power(t, 1 / expr.p.value) if t.is_nonneg() else np.inf