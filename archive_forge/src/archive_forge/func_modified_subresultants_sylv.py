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
def modified_subresultants_sylv(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x]. It is assumed
    that deg(f) >= deg(g).

    Computes the modified subresultant polynomial remainder sequence (prs)
    of f, g by evaluating determinants of appropriately selected
    submatrices of sylvester(f, g, x, 2). The dimensions of the
    latter are (2*deg(f)) x (2*deg(f)).

    Each coefficient is computed by evaluating the determinant of the
    corresponding submatrix of sylvester(f, g, x, 2).

    If the modified subresultant prs is complete, then the output coincides
    with the Sturmian sequence of the polynomials f, g.

    References:
    ===========
    1. A. G. Akritas,G.I. Malaschonok and P.S. Vigklas:
    Sturm Sequences and Modified Subresultant Polynomial Remainder
    Sequences. Serdica Journal of Computing, Vol. 8, No 1, 29--46, 2014.

    """
    if f == 0 or g == 0:
        return [f, g]
    n = degF = degree(f, x)
    m = degG = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, degF, degG, f, g = (m, n, degG, degF, g, f)
    if n > 0 and m == 0:
        return [f, g]
    SR_L = [f, g]
    S = sylvester(f, g, x, 2)
    j = m - 1
    while j > 0:
        Sp = S[0:2 * n - 2 * j, :]
        coeff_L, k, l = ([], Sp.rows, 0)
        while l <= j:
            coeff_L.append(Sp[:, 0:k].det())
            Sp.col_swap(k - 1, k + l)
            l += 1
        SR_L.append(Poly(coeff_L, x).as_expr())
        j -= 1
    SR_L.append(S.det())
    return process_matrix_output(SR_L, x)