from .plot import BaseSeries, Plot
from .experimental_lambdify import experimental_lambdify, vectorized_lambdify
from .intervalmath import interval
from sympy.core.relational import (Equality, GreaterThan, LessThan,
from sympy.core.containers import Tuple
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.logic.boolalg import BooleanFunction
from sympy.polys.polyutils import _sort_gens
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import flatten
import warnings
def get_raster(self):
    func = experimental_lambdify((self.var_x, self.var_y), self.expr, use_interval=True)
    xinterval = interval(self.start_x, self.end_x)
    yinterval = interval(self.start_y, self.end_y)
    try:
        func(xinterval, yinterval)
    except AttributeError:
        if self.use_interval_math:
            warnings.warn('Adaptive meshing could not be applied to the expression. Using uniform meshing.', stacklevel=7)
        self.use_interval_math = False
    if self.use_interval_math:
        return self._get_raster_interval(func)
    else:
        return self._get_meshes_grid()