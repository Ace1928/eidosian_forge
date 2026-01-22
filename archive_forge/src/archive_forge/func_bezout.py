from sympy.concrete.summations import summation
from sympy.core.function import expand
from sympy.core.numbers import nan
from sympy.core.singleton import S
from sympy.core.symbol import Dummy as var
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import eye, Matrix, zeros
from sympy.printing.pretty.pretty import pretty_print as pprint
from sympy.simplify.simplify import simplify
from sympy.polys.domains import QQ
from sympy.polys.polytools import degree, LC, Poly, pquo, quo, prem, rem
from sympy.polys.polyerrors import PolynomialError
def bezout(p, q, x, method='bz'):
    """
    The input polynomials p, q are in Z[x] or in Q[x]. Let
    mx = max(degree(p, x), degree(q, x)).

    The default option bezout(p, q, x, method='bz') returns Bezout's
    symmetric matrix of p and q, of dimensions (mx) x (mx). The
    determinant of this matrix is equal to the determinant of sylvester2,
    Sylvester's matrix of 1853, whose dimensions are (2*mx) x (2*mx);
    however the subresultants of these two matrices may differ.

    The other option, bezout(p, q, x, 'prs'), is of interest to us
    in this module because it returns a matrix equivalent to sylvester2.
    In this case all subresultants of the two matrices are identical.

    Both the subresultant polynomial remainder sequence (prs) and
    the modified subresultant prs of p and q can be computed by
    evaluating determinants of appropriately selected submatrices of
    bezout(p, q, x, 'prs') --- one determinant per coefficient of the
    remainder polynomials.

    The matrices bezout(p, q, x, 'bz') and bezout(p, q, x, 'prs')
    are related by the formula

    bezout(p, q, x, 'prs') =
    backward_eye(deg(p)) * bezout(p, q, x, 'bz') * backward_eye(deg(p)),

    where backward_eye() is the backward identity function.

    References
    ==========
    1. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants
    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,
    Vol. 15, 233-266, 2004.

    """
    m, n = (degree(Poly(p, x), x), degree(Poly(q, x), x))
    if m == n and n < 0:
        return Matrix([])
    if m == n and n == 0:
        return Matrix([])
    if m == 0 and n < 0:
        return Matrix([])
    elif m < 0 and n == 0:
        return Matrix([])
    if m >= 1 and n < 0:
        return Matrix([0])
    elif m < 0 and n >= 1:
        return Matrix([0])
    y = var('y')
    expr = p * q.subs({x: y}) - p.subs({x: y}) * q
    poly = Poly(quo(expr, x - y), x, y)
    mx = max(m, n)
    B = zeros(mx)
    for i in range(mx):
        for j in range(mx):
            if method == 'prs':
                B[mx - 1 - i, mx - 1 - j] = poly.nth(i, j)
            else:
                B[i, j] = poly.nth(i, j)
    return B