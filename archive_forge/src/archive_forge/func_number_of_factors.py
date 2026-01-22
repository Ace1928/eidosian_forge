import abc
import math
import functools
from numbers import Integral
from collections.abc import Iterable, MutableSequence
from enum import Enum
from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.core.base import ConcreteModel, Objective, maximize, minimize, Block
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.base.var import Var, IndexedVar
from pyomo.core.expr.numvalue import value, native_numeric_types
from pyomo.opt.results import check_optimal_termination
from pyomo.contrib.pyros.util import add_bounds_for_uncertain_parameters
@number_of_factors.setter
def number_of_factors(self, val):
    if hasattr(self, '_number_of_factors'):
        raise AttributeError("Attribute 'number_of_factors' is immutable")
    else:
        validate_arg_type('number_of_factors', val, Integral)
        if val < 1:
            raise ValueError(f"Attribute 'number_of_factors' must be a positive int (provided value {val})")
    self._number_of_factors = val