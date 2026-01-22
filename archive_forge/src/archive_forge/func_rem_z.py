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
def rem_z(p, q, x):
    """
    Intended mainly for p, q polynomials in Z[x] so that,
    on dividing p by q, the remainder will also be in Z[x]. (However,
    it also works fine for polynomials in Q[x].) It is assumed
    that degree(p, x) >= degree(q, x).

    It premultiplies p by the _absolute_ value of the leading coefficient
    of q, raised to the power deg(p) - deg(q) + 1 and then performs
    polynomial division in Q[x], using the function rem(p, q, x).

    By contrast the function prem(p, q, x) does _not_ use the absolute
    value of the leading coefficient of q.
    This results not only in ``messing up the signs'' of the Euclidean and
    Sturmian prs's as mentioned in the second reference,
    but also in violation of the main results of the first and third
    references --- Theorem 4 and Theorem 1 respectively. Theorems 4 and 1
    establish a one-to-one correspondence between the Euclidean and the
    Sturmian prs of p, q, on one hand, and the subresultant prs of p, q,
    on the other.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials.''
    Serdica Journal of Computing, 9(2) (2015), 123-138.

    2. https://planetMath.org/sturmstheorem

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result on
    the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    """
    if p.as_poly().is_univariate and q.as_poly().is_univariate and (p.as_poly().gens == q.as_poly().gens):
        delta = degree(p, x) - degree(q, x) + 1
        return rem(Abs(LC(q, x)) ** delta * p, q, x)
    else:
        return prem(p, q, x)