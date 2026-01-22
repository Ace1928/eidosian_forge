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
def test_max_shear_force():
    E = Symbol('E')
    I = Symbol('I')
    b = Beam(3, E, I)
    R, M = symbols('R, M')
    b.apply_load(R, 0, -1)
    b.apply_load(M, 0, -2)
    b.apply_load(2, 3, -1)
    b.apply_load(4, 2, -1)
    b.apply_load(2, 2, 0, end=3)
    b.solve_for_reaction_loads(R, M)
    assert b.max_shear_force() == (Interval(0, 2), 8)
    l = symbols('l', positive=True)
    P = Symbol('P')
    b = Beam(l, E, I)
    R1, R2 = symbols('R1, R2')
    b.apply_load(R1, 0, -1)
    b.apply_load(R2, l, -1)
    b.apply_load(P, 0, 0, end=l)
    b.solve_for_reaction_loads(R1, R2)
    assert b.max_shear_force() == (0, l * Abs(P) / 2)