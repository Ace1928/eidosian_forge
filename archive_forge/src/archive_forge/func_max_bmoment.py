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
def max_bmoment(self):
    """Returns maximum Shear force and its coordinate
        in the Beam object."""
    bending_curve = self.bending_moment()
    x = self.variable
    terms = bending_curve.args
    singularity = []
    for term in terms:
        if isinstance(term, Mul):
            term = term.args[-1]
        singularity.append(term.args[1])
    singularity.sort()
    singularity = list(set(singularity))
    intervals = []
    moment_values = []
    for i, s in enumerate(singularity):
        if s == 0:
            continue
        try:
            moment_slope = Piecewise((float('nan'), x <= singularity[i - 1]), (self.shear_force().rewrite(Piecewise), x < s), (float('nan'), True))
            points = solve(moment_slope, x)
            val = []
            for point in points:
                val.append(abs(bending_curve.subs(x, point)))
            points.extend([singularity[i - 1], s])
            val += [abs(limit(bending_curve, x, singularity[i - 1], '+')), abs(limit(bending_curve, x, s, '-'))]
            max_moment = max(val)
            moment_values.append(max_moment)
            intervals.append(points[val.index(max_moment)])
        except NotImplementedError:
            initial_moment = limit(bending_curve, x, singularity[i - 1], '+')
            final_moment = limit(bending_curve, x, s, '-')
            if bending_curve.subs(x, (singularity[i - 1] + s) / 2) == (initial_moment + final_moment) / 2 and initial_moment != final_moment:
                moment_values.extend([initial_moment, final_moment])
                intervals.extend([singularity[i - 1], s])
            else:
                moment_values.append(final_moment)
                intervals.append(Interval(singularity[i - 1], s))
    moment_values = list(map(abs, moment_values))
    maximum_moment = max(moment_values)
    point = intervals[moment_values.index(maximum_moment)]
    return (point, maximum_moment)