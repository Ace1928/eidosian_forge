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
def subresultants_vv_2(p, q, x):
    """
    p, q are polynomials in Z[x] (intended) or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p, q by triangularizing,
    in Z[x] or in Q[x], all the smaller matrices encountered in the
    process of triangularizing sylvester2, Sylvester's matrix of 1853;
    see references 1 and 2 for Van Vleck's method.

    If the sylvester2 matrix has big dimensions use this version,
    where sylvester2 is used implicitly. If you want to see the final,
    triangularized matrix sylvester2, then use the first version,
    subresultants_vv(p, q, x, 1).

    sylvester1, Sylvester's matrix of 1840, is also used to compute
    one subresultant per remainder; namely, that of the leading
    coefficient, in order to obtain the correct sign and to
    ``force'' the remainder coefficients to become subresultants.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G.: ``A new method for computing polynomial greatest
    common divisors and polynomial remainder sequences.''
    Numerische MatheMatik 52, 119-127, 1988.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem
    by Van Vleck Regarding Sturm Sequences.''
    Serdica Journal of Computing, 7, No 4, 101-134, 2013.

    3. Akritas, A. G.:``Three New Methods for Computing Subresultant
    Polynomial Remainder Sequences (PRS's).'' Serdica Journal of Computing 9(1) (2015), 1-26.

    """
    if p == 0 or q == 0:
        return [p, q]
    f, g = (p, q)
    n = deg_f = degree(f, x)
    m = deg_g = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, deg_f, deg_g, f, g = (m, n, deg_g, deg_f, g, f)
    if n > 0 and m == 0:
        return [f, g]
    s1 = sylvester(f, g, x, 1)
    sr_list = [f, g]
    col_num = 2 * n
    row0 = Poly(f, x, domain=QQ).all_coeffs()
    leng0 = len(row0)
    for i in range(col_num - leng0):
        row0.append(0)
    row0 = Matrix([row0])
    row1 = Poly(g, x, domain=QQ).all_coeffs()
    leng1 = len(row1)
    for i in range(col_num - leng1):
        row1.append(0)
    row1 = Matrix([row1])
    while deg_g > 0:
        M = create_ma(deg_f, deg_g, row1, row0, col_num)
        for i in range(deg_f - deg_g + 1):
            M1 = pivot(M, i, i)
            M = M1[:, :]
        d = find_degree(M, deg_f)
        if d is None:
            return sr_list
        exp_deg = deg_g - 1
        sign_value = correct_sign(n, m, s1, exp_deg, exp_deg - d)
        poly = row2poly(M[M.rows - 1, :], d, x)
        poly = simplify(poly / LC(poly, x) * sign_value)
        sr_list.append(poly)
        deg_f, deg_g = (deg_g, d)
        row0 = row1
        row1 = Poly(poly, x, domain=QQ).all_coeffs()
        leng1 = len(row1)
        for i in range(col_num - leng1):
            row1.append(0)
        row1 = Matrix([row1])
    return sr_list