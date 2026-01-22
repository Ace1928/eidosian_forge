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
def process_matrix_output(poly_seq, x):
    """
    poly_seq is a polynomial remainder sequence computed either by
    (modified_)subresultants_bezout or by (modified_)subresultants_sylv.

    This function removes from poly_seq all zero polynomials as well
    as all those whose degree is equal to the degree of a preceding
    polynomial in poly_seq, as we scan it from left to right.

    """
    L = poly_seq[:]
    d = degree(L[1], x)
    i = 2
    while i < len(L):
        d_i = degree(L[i], x)
        if d_i < 0:
            L.remove(L[i])
            i = i - 1
        if d == d_i:
            L.remove(L[i])
            i = i - 1
        if d_i >= 0:
            d = d_i
        i = i + 1
    return L