import string
from ..sage_helper import _within_sage, sage_method
def fast_determinant_of_laurent_poly_matrix(A):
    """
    Return the determinant of the given matrix up to
    a power of t^n, using the faster algorithm for
    polynomial entries.
    """
    R = A.base_ring()
    minexp = minimum_exponents(A.list())
    P = R.polynomial_ring()
    MS = A.parent().change_ring(P)
    Ap = MS([convert_laurent_to_poly(p, minexp, P) for p in A.list()])
    return Ap.det()