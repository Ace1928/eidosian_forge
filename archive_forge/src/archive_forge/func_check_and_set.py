from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import arity, Function
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.printing.latex import latex
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from .experimental_lambdify import (vectorized_lambdify, lambdify)
from sympy.plotting.textplot import textplot
def check_and_set(t_name, t):
    if t:
        if not is_real(t):
            raise ValueError('All numbers from {}={} must be real'.format(t_name, t))
        if not is_finite(t):
            raise ValueError('All numbers from {}={} must be finite'.format(t_name, t))
        setattr(self, t_name, (float(t[0]), float(t[1])))