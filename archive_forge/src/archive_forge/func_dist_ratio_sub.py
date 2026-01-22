from cvxpy import atoms
from cvxpy.atoms.affine import binary_operators as bin_op
from cvxpy.atoms.affine.diag import diag_vec
from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variable import Variable
def dist_ratio_sub(expr, t):
    x = expr.args[0]
    a = expr.a
    b = expr.b

    def sublevel_set():
        if t.value > 1:
            return False
        tsq = t.value ** 2
        return (1 - tsq ** 2) * atoms.sum_squares(x) - atoms.matmul(2 * (a - tsq * b), x) + atoms.sum_squares(a) - tsq * atoms.sum_squares(b) <= 0
    return [sublevel_set]