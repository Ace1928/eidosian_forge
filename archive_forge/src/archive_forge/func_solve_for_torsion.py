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
def solve_for_torsion(self):
    """
        Solves for the angular deflection due to the torsional effects of
        moments being applied in the x-direction i.e. out of or into the beam.

        Here, a positive torque means the direction of the torque is positive
        i.e. out of the beam along the beam-axis. Likewise, a negative torque
        signifies a torque into the beam cross-section.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(20, E, G, I, A, x)
        >>> b.apply_moment_load(4, 4, -2, dir='x')
        >>> b.apply_moment_load(4, 8, -2, dir='x')
        >>> b.apply_moment_load(4, 8, -2, dir='x')
        >>> b.solve_for_torsion()
        >>> b.angular_deflection().subs(x, 3)
        18/(G*I)
        """
    x = self.variable
    sum_moments = 0
    for point in list(self._torsion_moment):
        sum_moments += self._torsion_moment[point]
    list(self._torsion_moment).sort()
    pointsList = list(self._torsion_moment)
    torque_diagram = Piecewise((sum_moments, x <= pointsList[0]), (0, x >= pointsList[0]))
    for i in range(len(pointsList))[1:]:
        sum_moments -= self._torsion_moment[pointsList[i - 1]]
        torque_diagram += Piecewise((0, x <= pointsList[i - 1]), (sum_moments, x <= pointsList[i]), (0, x >= pointsList[i]))
    integrated_torque_diagram = integrate(torque_diagram)
    self._angular_deflection = integrated_torque_diagram / (self.shear_modulus * self.polar_moment())