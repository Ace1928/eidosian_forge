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
def test_parabolic_loads():
    E, I, L = symbols('E, I, L', positive=True, real=True)
    R, M, P = symbols('R, M, P', real=True)
    beam = Beam(L, E, I)
    beam.bc_deflection.append((0, 0))
    beam.bc_slope.append((0, 0))
    beam.apply_load(R, 0, -1)
    beam.apply_load(M, 0, -2)
    beam.apply_load(1, 0, 2)
    beam.solve_for_reaction_loads(R, M)
    assert beam.reaction_loads[R] == -L ** 3 / 3
    beam = Beam(2 * L, E, I)
    beam.bc_deflection.append((0, 0))
    beam.bc_slope.append((0, 0))
    beam.apply_load(R, 0, -1)
    beam.apply_load(M, 0, -2)
    beam.apply_load(1, 0, 2, end=L)
    beam.solve_for_reaction_loads(R, M)
    assert beam.reaction_loads[R] == -L ** 3 / 3
    beam = Beam(2 * L, E, I)
    beam.apply_load(P, 0, 0, end=L)
    loading = beam.load.xreplace({L: 10, E: 20, I: 30, P: 40})
    assert loading.xreplace({x: 5}) == 40
    assert loading.xreplace({x: 15}) == 0
    beam = Beam(2 * L, E, I)
    beam.apply_load(P, 0, 1, end=L)
    assert beam.load == P * SingularityFunction(x, 0, 1) - P * SingularityFunction(x, L, 1) - P * L * SingularityFunction(x, L, 0)
    beam = Beam(2 * L, E, I)
    beam.apply_load(P, 0, 8, end=L)
    loading = beam.load.xreplace({L: 10, E: 20, I: 30, P: 40})
    assert loading.xreplace({x: 5}) == 40 * 5 ** 8
    assert loading.xreplace({x: 15}) == 0