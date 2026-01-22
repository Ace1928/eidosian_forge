import warnings
from ase.units import kJ
import numpy as np
from scipy.optimize import curve_fit
def check_birchmurnaghan():
    from sympy import symbols, Rational, diff, simplify
    v, b, bp, v0 = symbols('v b bp v0')
    x = (v0 / v) ** Rational(2, 3)
    e = 9 * b * v0 * (x - 1) ** 2 * (6 + bp * (x - 1) - 4 * x) / 16
    print(e)
    B = diff(e, v, 2) * v
    BP = -v * diff(B, v) / b
    print(simplify(B.subs(v, v0)))
    print(simplify(BP.subs(v, v0)))