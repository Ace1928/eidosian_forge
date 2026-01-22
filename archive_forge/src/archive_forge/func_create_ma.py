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
def create_ma(deg_f, deg_g, row1, row2, col_num):
    """
    Creates a ``small'' matrix M to be triangularized.

    deg_f, deg_g are the degrees of the divident and of the
    divisor polynomials respectively, deg_g > deg_f.

    The coefficients of the divident poly are the elements
    in row2 and those of the divisor poly are the elements
    in row1.

    col_num defines the number of columns of the matrix M.

    """
    if deg_g - deg_f >= 1:
        print('Reverse degrees')
        return
    m = zeros(deg_f - deg_g + 2, col_num)
    for i in range(deg_f - deg_g + 1):
        m[i, :] = rotate_r(row1, i)
    m[deg_f - deg_g + 1, :] = row2
    return m