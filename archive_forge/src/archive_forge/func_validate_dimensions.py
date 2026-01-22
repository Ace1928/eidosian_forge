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
def validate_dimensions(arr_name, arr, dim, display_value=False):
    """
    Validate dimension of an array-like object.
    Raise Exception if validation fails.
    """
    if is_ragged(arr):
        raise ValueError(f'Argument `{arr_name}` should not be a ragged array-like (nested sequence of lists, tuples, arrays of different shape)')
    array = np.asarray(arr)
    if len(array.shape) != dim:
        val_str = f' from provided value {str(arr)}' if display_value else ''
        raise ValueError(f'Argument `{arr_name}` must be a {dim}-dimensional array-like (detected {len(array.shape)} dimensions{val_str})')
    elif array.shape[-1] == 0:
        raise ValueError(f'Last dimension of argument `{arr_name}` must be non-empty (detected shape {array.shape})')