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
def test_variable_moment():
    E = Symbol('E')
    I = Symbol('I')
    b = Beam(4, E, 2 * (4 - x))
    b.apply_load(20, 4, -1)
    R, M = symbols('R, M')
    b.apply_load(R, 0, -1)
    b.apply_load(M, 0, -2)
    b.bc_deflection = [(0, 0)]
    b.bc_slope = [(0, 0)]
    b.solve_for_reaction_loads(R, M)
    assert b.slope().expand() == ((10 * x * SingularityFunction(x, 0, 0) - 10 * (x - 4) * SingularityFunction(x, 4, 0)) / E).expand()
    assert b.deflection().expand() == ((5 * x ** 2 * SingularityFunction(x, 0, 0) - 10 * Piecewise((0, Abs(x) / 4 < 1), (16 * meijerg(((3, 1), ()), ((), (2, 0)), x / 4), True)) + 40 * SingularityFunction(x, 4, 1)) / E).expand()
    b = Beam(4, E - x, I)
    b.apply_load(20, 4, -1)
    R, M = symbols('R, M')
    b.apply_load(R, 0, -1)
    b.apply_load(M, 0, -2)
    b.bc_deflection = [(0, 0)]
    b.bc_slope = [(0, 0)]
    b.solve_for_reaction_loads(R, M)
    assert b.slope().expand() == ((-80 * (-log(-E) + log(-E + x)) * SingularityFunction(x, 0, 0) + 80 * (-log(-E + 4) + log(-E + x)) * SingularityFunction(x, 4, 0) + 20 * (-E * log(-E) + E * log(-E + x) + x) * SingularityFunction(x, 0, 0) - 20 * (-E * log(-E + 4) + E * log(-E + x) + x - 4) * SingularityFunction(x, 4, 0)) / I).expand()