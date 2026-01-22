from typing import Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.atom import Atom
def diff_pos(x, y):
    """The difference :math:`x - y` with domain `\\{x, y : x > y > 0\\}`.

    This atom is log-log concave.

    Parameters
    ----------
    x : :class:`~cvxpy.expressions.expression.Expression`
        An Expression.
    y : :class:`~cvxpy.expressions.expression.Expression`
        An Expression.
    """
    return multiply(x, one_minus_pos(y / x))