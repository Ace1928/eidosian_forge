from sympy.core import S, Symbol, diff, symbols
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.sympify import sympify
from sympy.solvers import linsolve
from sympy.solvers.ode.ode import dsolve
from sympy.solvers.solvers import solve
from sympy.printing import sstr
from sympy.functions import SingularityFunction, Piecewise, factorial
from sympy.integrals import integrate
from sympy.series import limit
from sympy.plotting import plot, PlotGrid
from sympy.geometry.entity import GeometryEntity
from sympy.external import import_module
from sympy.sets.sets import Interval
from sympy.utilities.lambdify import lambdify
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import iterable
def solve_slope_deflection(self):
    x = self.variable
    l = self.length
    E = self.elastic_modulus
    G = self.shear_modulus
    I = self.second_moment
    if isinstance(I, list):
        I_y, I_z = (I[0], I[1])
    else:
        I_y = I_z = I
    A = self._area
    load = self._load_vector
    moment = self._moment_load_vector
    defl = Function('defl')
    theta = Function('theta')
    eq = Derivative(E * A * Derivative(defl(x), x), x) + load[0]
    def_x = dsolve(Eq(eq, 0), defl(x)).args[1]
    C1 = Symbol('C1')
    C2 = Symbol('C2')
    constants = list(linsolve([def_x.subs(x, 0), def_x.subs(x, l)], C1, C2).args[0])
    def_x = def_x.subs({C1: constants[0], C2: constants[1]})
    slope_x = def_x.diff(x)
    self._deflection[0] = def_x
    self._slope[0] = slope_x
    C_i = Symbol('C_i')
    eq1 = Derivative(E * I_z * Derivative(theta(x), x), x) + (integrate(-load[1], x) + C_i) + moment[2]
    slope_z = dsolve(Eq(eq1, 0)).args[1]
    constants = list(linsolve([slope_z.subs(x, 0), slope_z.subs(x, l)], C1, C2).args[0])
    slope_z = slope_z.subs({C1: constants[0], C2: constants[1]})
    eq2 = G * A * Derivative(defl(x), x) + load[1] * x - C_i - G * A * slope_z
    def_y = dsolve(Eq(eq2, 0), defl(x)).args[1]
    constants = list(linsolve([def_y.subs(x, 0), def_y.subs(x, l)], C1, C_i).args[0])
    self._deflection[1] = def_y.subs({C1: constants[0], C_i: constants[1]})
    self._slope[2] = slope_z.subs(C_i, constants[1])
    eq1 = Derivative(E * I_y * Derivative(theta(x), x), x) + (integrate(load[2], x) - C_i) + moment[1]
    slope_y = dsolve(Eq(eq1, 0)).args[1]
    constants = list(linsolve([slope_y.subs(x, 0), slope_y.subs(x, l)], C1, C2).args[0])
    slope_y = slope_y.subs({C1: constants[0], C2: constants[1]})
    eq2 = G * A * Derivative(defl(x), x) + load[2] * x - C_i + G * A * slope_y
    def_z = dsolve(Eq(eq2, 0)).args[1]
    constants = list(linsolve([def_z.subs(x, 0), def_z.subs(x, l)], C1, C_i).args[0])
    self._deflection[2] = def_z.subs({C1: constants[0], C_i: constants[1]})
    self._slope[1] = slope_y.subs(C_i, constants[1])