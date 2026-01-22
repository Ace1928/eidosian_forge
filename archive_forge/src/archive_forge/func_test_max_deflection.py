from sympy.core.function import expand
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.sets.sets import Interval
from sympy.simplify.simplify import simplify
from sympy.physics.continuum_mechanics.beam import Beam
from sympy.functions import SingularityFunction, Piecewise, meijerg, Abs, log
from sympy.testing.pytest import raises
from sympy.physics.units import meter, newton, kilo, giga, milli
from sympy.physics.continuum_mechanics.beam import Beam3D
from sympy.geometry import Circle, Polygon, Point2D, Triangle
from sympy.core.sympify import sympify
def test_max_deflection():
    E, I, l, F = symbols('E, I, l, F', positive=True)
    b = Beam(l, E, I)
    b.bc_deflection = [(0, 0), (l, 0)]
    b.bc_slope = [(0, 0), (l, 0)]
    b.apply_load(F / 2, 0, -1)
    b.apply_load(-F * l / 8, 0, -2)
    b.apply_load(F / 2, l, -1)
    b.apply_load(F * l / 8, l, -2)
    b.apply_load(-F, l / 2, -1)
    assert b.max_deflection() == (l / 2, F * l ** 3 / (192 * E * I))