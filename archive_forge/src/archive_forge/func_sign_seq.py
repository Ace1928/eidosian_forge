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
def sign_seq(poly_seq, x):
    """
    Given a sequence of polynomials poly_seq, it returns
    the sequence of signs of the leading coefficients of
    the polynomials in poly_seq.

    """
    return [sign(LC(poly_seq[i], x)) for i in range(len(poly_seq))]