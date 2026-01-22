from sympy.core.symbol import symbols
from sympy.functions.special.spherical_harmonics import Ynm
def timeit_Ynm_xy():
    Ynm(1, 1, x, y)