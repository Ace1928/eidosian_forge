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
def generate_shape_str(shape, required_shape):
    shape_str = ''
    assert len(shape) == len(required_shape)
    for idx, (sval, rsval) in enumerate(zip(shape, required_shape)):
        if rsval is None:
            shape_str += '...'
        else:
            shape_str += f'{sval}'
        if idx < len(shape) - 1:
            shape_str += ','
    return '(' + shape_str + ')'