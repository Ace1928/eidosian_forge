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
def plot_ild_reactions(self, subs=None):
    """

        Plots the Influence Line Diagram of Reaction Forces
        under the effect of a moving load. This function
        should be called after calling solve_for_ild_reactions().

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Examples
        ========

        There is a beam of length 10 meters. A point load of magnitude 5KN
        is also applied from top of the beam, at a distance of 4 meters
        from the starting point. There are two simple supports below the
        beam, located at the starting point and at a distance of 7 meters
        from the starting point. Plot the I.L.D. equations for reactions
        at both support points under the effect of a moving load
        of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_7 = symbols('R_0, R_7')
            >>> b = Beam(10, E, I)
            >>> b.apply_support(0, 'roller')
            >>> b.apply_support(7, 'roller')
            >>> b.apply_load(5,4,-1)
            >>> b.solve_for_ild_reactions(1,R_0,R_7)
            >>> b.ild_reactions
            {R_0: x/7 - 22/7, R_7: -x/7 - 20/7}
            >>> b.plot_ild_reactions()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: x/7 - 22/7 for x over (0.0, 10.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -x/7 - 20/7 for x over (0.0, 10.0)

        """
    if not self._ild_reactions:
        raise ValueError('I.L.D. reaction equations not found. Please use solve_for_ild_reactions() to generate the I.L.D. reaction equations.')
    x = self.variable
    ildplots = []
    if subs is None:
        subs = {}
    for reaction in self._ild_reactions:
        for sym in self._ild_reactions[reaction].atoms(Symbol):
            if sym != x and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
    for sym in self._length.atoms(Symbol):
        if sym != x and sym not in subs:
            raise ValueError('Value of %s was not passed.' % sym)
    for reaction in self._ild_reactions:
        ildplots.append(plot(self._ild_reactions[reaction].subs(subs), (x, 0, self._length.subs(subs)), title='I.L.D. for Reactions', xlabel=x, ylabel=reaction, line_color='blue', show=False))
    return PlotGrid(len(ildplots), 1, *ildplots)