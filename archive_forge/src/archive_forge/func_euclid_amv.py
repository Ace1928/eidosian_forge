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
def euclid_amv(f, g, x):
    """
    f, g are polynomials in Z[x] or Q[x]. It is assumed
    that degree(f, x) >= degree(g, x).

    Computes the Euclidean sequence of p and q in Z[x] or Q[x].

    If the Euclidean sequence is complete the coefficients of the polynomials
    in the sequence are subresultants. That is, they are  determinants of
    appropriately selected submatrices of sylvester1, Sylvester's matrix of 1840.
    In this case the Euclidean sequence coincides with the subresultant prs,
    of the polynomials p, q.

    If the Euclidean sequence is incomplete the signs of the coefficients of the
    polynomials in the sequence may differ from the signs of the coefficients of
    the corresponding polynomials in the subresultant prs; however, the absolute
    values are the same.

    To compute the coefficients, no determinant evaluation takes place.
    Instead, polynomial divisions in Z[x] or Q[x] are performed, using
    the function rem_z(f, g, x);  the coefficients of the remainders
    computed this way become subresultants with the help of the
    Collins-Brown-Traub formula for coefficient reduction.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.

    """
    if f == 0 or g == 0:
        return [f, g]
    d0 = degree(f, x)
    d1 = degree(g, x)
    if d0 == 0 and d1 == 0:
        return [f, g]
    if d1 > d0:
        d0, d1 = (d1, d0)
        f, g = (g, f)
    if d0 > 0 and d1 == 0:
        return [f, g]
    a0 = f
    a1 = g
    euclid_seq = [a0, a1]
    deg_dif_p1, c = (degree(a0, x) - degree(a1, x) + 1, -1)
    i = 1
    a2 = rem_z(a0, a1, x) / Abs((-1) ** deg_dif_p1)
    euclid_seq.append(a2)
    d2 = degree(a2, x)
    while d2 >= 1:
        a0, a1, d0, d1 = (a1, a2, d1, d2)
        i += 1
        sigma0 = -LC(a0)
        c = sigma0 ** (deg_dif_p1 - 1) / c ** (deg_dif_p1 - 2)
        deg_dif_p1 = degree(a0, x) - d2 + 1
        a2 = rem_z(a0, a1, x) / Abs(c ** (deg_dif_p1 - 1) * sigma0)
        euclid_seq.append(a2)
        d2 = degree(a2, x)
    m = len(euclid_seq)
    if euclid_seq[m - 1] == nan or euclid_seq[m - 1] == 0:
        euclid_seq.pop(m - 1)
    return euclid_seq