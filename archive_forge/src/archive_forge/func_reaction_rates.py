from collections import OrderedDict
from functools import reduce, partial
from itertools import chain
from operator import attrgetter, mul
import math
import warnings
from ..units import (
from ..util.pyutil import deprecated
from ..util._expr import Expr, Symbol
from .rates import RateExpr, MassAction
def reaction_rates(t, y, p, backend=math):
    variables = dict(chain(y.items(), p.items()))
    if 'time' in variables:
        raise ValueError("Key 'time' is reserved.")
    variables['time'] = t
    for k, act in _active_subst.items():
        if unit_registry is not None and act.args:
            _, act = act.dedimensionalisation(unit_registry)
        variables[k] = act(variables, backend=backend)
    variables.update(_passive_subst)
    return [ratex(variables, backend=backend, reaction=rxn) for rxn, ratex in zip(rsys.rxns, r_exprs)]