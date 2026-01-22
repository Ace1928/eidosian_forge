from sympy.core import Symbol
from sympy.core.evalf import N
from sympy.core.numbers import I, Rational
from sympy.functions import sqrt
from sympy.polys.polytools import Poly
from sympy.utilities import public
def l0(self, theta):
    F = self.F
    a = self.a
    l0 = Poly(a, x).eval(theta) / F
    return l0