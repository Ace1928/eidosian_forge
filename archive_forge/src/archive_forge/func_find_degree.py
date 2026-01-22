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
def find_degree(M, deg_f):
    """
    Finds the degree of the poly corresponding (after triangularization)
    to the _last_ row of the ``small'' matrix M, created by create_ma().

    deg_f is the degree of the divident poly.
    If _last_ row is all 0's returns None.

    """
    j = deg_f
    for i in range(0, M.cols):
        if M[M.rows - 1, i] == 0:
            j = j - 1
        else:
            return j if j >= 0 else 0