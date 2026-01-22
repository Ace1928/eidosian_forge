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
def test_cross_section():
    I = Symbol('I')
    l = Symbol('l')
    E = Symbol('E')
    C3, C4 = symbols('C3, C4')
    a, c, g, h, r, n = symbols('a, c, g, h, r, n')
    b0 = Beam(l, E, I)
    assert b0.second_moment == I
    assert b0.cross_section == None
    b0.cross_section = Circle((0, 0), 5)
    assert b0.second_moment == pi * Rational(625, 4)
    assert b0.cross_section == Circle((0, 0), 5)
    b0.second_moment = 2 * n - 6
    assert b0.second_moment == 2 * n - 6
    assert b0.cross_section == None
    with raises(ValueError):
        b0.second_moment = Circle((0, 0), 5)
    b1 = Beam(50, E, Circle((0, 0), r))
    assert b1.cross_section == Circle((0, 0), r)
    assert b1.second_moment == pi * r * Abs(r) ** 3 / 4
    b1.apply_load(-10, 0, -1)
    b1.apply_load(R1, 5, -1)
    b1.apply_load(R2, 50, -1)
    b1.apply_load(90, 45, -2)
    b1.solve_for_reaction_loads(R1, R2)
    assert b1.load == -10 * SingularityFunction(x, 0, -1) + 82 * SingularityFunction(x, 5, -1) / S(9) + 90 * SingularityFunction(x, 45, -2) + 8 * SingularityFunction(x, 50, -1) / 9
    assert b1.bending_moment() == 10 * SingularityFunction(x, 0, 1) - 82 * SingularityFunction(x, 5, 1) / 9 - 90 * SingularityFunction(x, 45, 0) - 8 * SingularityFunction(x, 50, 1) / 9
    q = (-5 * SingularityFunction(x, 0, 2) + 41 * SingularityFunction(x, 5, 2) / S(9) + 90 * SingularityFunction(x, 45, 1) + 4 * SingularityFunction(x, 50, 2) / S(9)) / (pi * E * r * Abs(r) ** 3)
    assert b1.slope() == C3 + 4 * q
    q = (-5 * SingularityFunction(x, 0, 3) / 3 + 41 * SingularityFunction(x, 5, 3) / 27 + 45 * SingularityFunction(x, 45, 2) + 4 * SingularityFunction(x, 50, 3) / 27) / (pi * E * r * Abs(r) ** 3)
    assert b1.deflection() == C3 * x + C4 + 4 * q
    b2 = Beam(20, E, Polygon((0, 0), (a, 0), (a, c), (0, c)))
    assert b2.cross_section == Polygon((0, 0), (a, 0), (a, c), (0, c))
    assert b2.second_moment == a * c ** 3 / 12
    b3 = Beam(15, E, Triangle((0, 0), (g, 0), (g / 2, h)))
    assert b3.cross_section == Triangle(Point2D(0, 0), Point2D(g, 0), Point2D(g / 2, h))
    assert b3.second_moment == g * h ** 3 / 36
    b = b2.join(b3, 'fixed')
    b.apply_load(-30, 0, -1)
    b.apply_load(65, 0, -2)
    b.apply_load(40, 0, -1)
    b.bc_slope = [(0, 0)]
    b.bc_deflection = [(0, 0)]
    assert b.second_moment == Piecewise((a * c ** 3 / 12, x <= 20), (g * h ** 3 / 36, x <= 35))
    assert b.cross_section == None
    assert b.length == 35
    assert b.slope().subs(x, 7) == 8400 / (E * a * c ** 3)
    assert b.slope().subs(x, 25) == 52200 / (E * g * h ** 3) + 39600 / (E * a * c ** 3)
    assert b.deflection().subs(x, 30) == -537000 / (E * g * h ** 3) - 712000 / (E * a * c ** 3)