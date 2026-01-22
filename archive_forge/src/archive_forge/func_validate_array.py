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
def validate_array(arr, arr_name, dim, valid_types, valid_type_desc=None, required_shape=None):
    """
    Validate shape and entry types of an array-like object.

    Parameters
    ----------
    arr : array_like
        Object to validate.
    arr_name : str
        A name/descriptor of the object to validate.
        Usually, this is the name of an object attribute
        to which the array is meant to be set.
    dim : int
        Required dimension of the array-like object.
    valid_types : set[type]
        Allowable type(s) for each entry of the array.
    valid_type_desc : str or None, optional
        Descriptor for the allowable types.
    required_shape : list or None, optional
        Specification of the length of the array in each dimension.
        If `None` is provided, no specifications are imposed.
        If a `list` is provided, then each entry of the list must be
        an `int` specifying the required length in the dimension
        corresponding to the position of the entry
        or `None` (meaning no requirement for the length in the
        corresponding dimension).
    """
    np_arr = np.array(arr, dtype=object)
    validate_dimensions(arr_name, np_arr, dim, display_value=False)

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
    if required_shape is not None:
        assert len(required_shape) == dim
        for idx, size in enumerate(required_shape):
            if size is not None and size != np_arr.shape[idx]:
                req_shape_str = generate_shape_str(required_shape, required_shape)
                actual_shape_str = generate_shape_str(np_arr.shape, required_shape)
                raise ValueError(f"Attribute '{arr_name}' should be of shape {req_shape_str}, but detected shape {actual_shape_str}")
    for val in np_arr.flat:
        validate_arg_type(arr_name, val, valid_types, valid_type_desc=valid_type_desc, is_entry_of_arg=True)