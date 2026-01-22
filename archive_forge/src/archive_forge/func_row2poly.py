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
def row2poly(row, deg, x):
    """
    Converts the row of a matrix to a poly of degree deg and variable x.
    Some entries at the beginning and/or at the end of the row may be zero.

    """
    k = 0
    poly = []
    leng = len(row)
    while row[k] == 0:
        k = k + 1
    for j in range(deg + 1):
        if k + j <= leng:
            poly.append(row[k + j])
    return Poly(poly, x)