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
def test_max_deflection_Beam3D():
    x = symbols('x')
    b = Beam3D(20, 40, 21, 100, 25)
    b.apply_load(15, start=0, order=0, dir='z')
    b.apply_load(12 * x, start=0, order=0, dir='y')
    b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
    b.solve_slope_deflection()
    c = sympify('495/14')
    p = sympify('-10 + 10*sqrt(10793)/43')
    q = sympify('(10 - 10*sqrt(10793)/43)**3/160 - 20/7 + (10 - 10*sqrt(10793)/43)**4/6400 + 20*sqrt(10793)/301 + 27*(10 - 10*sqrt(10793)/43)**2/560')
    assert b.max_deflection() == [(0, 0), (10, c), (p, q)]