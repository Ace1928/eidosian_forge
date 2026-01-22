from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import gcd
from sympy.sets.sets import Complement
from sympy.core import Basic, Tuple, diff, expand, Eq, Integer
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol
from sympy.solvers import solveset, nonlinsolve, diophantine
from sympy.polys import total_degree
from sympy.geometry import Point
from sympy.ntheory.factor_ import core
def regular_point(self):
    """
        Returns a point on the implicit region.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy.vector import ImplicitRegion
        >>> circle = ImplicitRegion((x, y), (x + 2)**2 + (y - 3)**2 - 16)
        >>> circle.regular_point()
        (-2, -1)
        >>> parabola = ImplicitRegion((x, y), x**2 - 4*y)
        >>> parabola.regular_point()
        (0, 0)
        >>> r = ImplicitRegion((x, y, z), (x + y + z)**4)
        >>> r.regular_point()
        (-10, -10, 20)

        References
        ==========

        - Erik Hillgarter, "Rational Points on Conics", Diploma Thesis, RISC-Linz,
          J. Kepler Universitat Linz, 1996. Available:
          https://www3.risc.jku.at/publications/download/risc_1355/Rational%20Points%20on%20Conics.pdf

        """
    equation = self.equation
    if len(self.variables) == 1:
        return (list(solveset(equation, self.variables[0], domain=S.Reals))[0],)
    elif len(self.variables) == 2:
        if self.degree == 2:
            coeffs = a, b, c, d, e, f = conic_coeff(self.variables, equation)
            if b ** 2 == 4 * a * c:
                x_reg, y_reg = self._regular_point_parabola(*coeffs)
            else:
                x_reg, y_reg = self._regular_point_ellipse(*coeffs)
            return (x_reg, y_reg)
    if len(self.variables) == 3:
        x, y, z = self.variables
        for x_reg in range(-10, 10):
            for y_reg in range(-10, 10):
                if not solveset(equation.subs({x: x_reg, y: y_reg}), self.variables[2], domain=S.Reals).is_empty:
                    return (x_reg, y_reg, list(solveset(equation.subs({x: x_reg, y: y_reg})))[0])
    if len(self.singular_points()) != 0:
        return list[self.singular_points()][0]
    raise NotImplementedError()