from cvxpy.atoms.affine.imag import imag
from cvxpy.atoms.affine.real import real
def param_canon(expr, real_args, imag_args, real2imag):
    if expr.is_real():
        return (expr, None)
    elif expr.is_imag():
        return (None, imag(expr))
    else:
        return (real(expr), imag(expr))