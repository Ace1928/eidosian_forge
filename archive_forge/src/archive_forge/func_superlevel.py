from cvxpy import atoms
from cvxpy.atoms.affine import binary_operators as bin_op
from cvxpy.atoms.affine.diag import diag_vec
from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variable import Variable
def superlevel(expr, t):
    """Return the t-level superlevel set for `expr`.

    Returned as a constraint phi_t(x) >= 0, where phi_t(x) is concave.
    """
    try:
        return SUPERLEVEL_SETS[type(expr)](expr, t)
    except KeyError:
        raise RuntimeError(f'The {type(expr)} atom is not yet supported in DQCP. Please file an issue here: https://github.com/cvxpy/cvxpy/issues')