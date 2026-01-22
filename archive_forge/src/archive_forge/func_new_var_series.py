import abc
import dataclasses
import math
import numbers
import typing
from typing import Callable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy import typing as npt
import pandas as pd
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver.python import model_builder_helper as mbh
from ortools.linear_solver.python import model_builder_numbers as mbn
def new_var_series(self, name: str, index: pd.Index, lower_bounds: Union[NumberT, pd.Series]=-math.inf, upper_bounds: Union[NumberT, pd.Series]=math.inf, is_integral: Union[bool, pd.Series]=False) -> pd.Series:
    """Creates a series of (scalar-valued) variables with the given name.

        Args:
          name (str): Required. The name of the variable set.
          index (pd.Index): Required. The index to use for the variable set.
          lower_bounds (Union[int, float, pd.Series]): Optional. A lower bound for
            variables in the set. If a `pd.Series` is passed in, it will be based on
            the corresponding values of the pd.Series. Defaults to -inf.
          upper_bounds (Union[int, float, pd.Series]): Optional. An upper bound for
            variables in the set. If a `pd.Series` is passed in, it will be based on
            the corresponding values of the pd.Series. Defaults to +inf.
          is_integral (bool, pd.Series): Optional. Indicates if the variable can
            only take integer values. If a `pd.Series` is passed in, it will be
            based on the corresponding values of the pd.Series. Defaults to False.

        Returns:
          pd.Series: The variable set indexed by its corresponding dimensions.

        Raises:
          TypeError: if the `index` is invalid (e.g. a `DataFrame`).
          ValueError: if the `name` is not a valid identifier or already exists.
          ValueError: if the `lowerbound` is greater than the `upperbound`.
          ValueError: if the index of `lower_bound`, `upper_bound`, or `is_integer`
          does not match the input index.
        """
    if not isinstance(index, pd.Index):
        raise TypeError('Non-index object is used as index')
    if not name.isidentifier():
        raise ValueError('name={} is not a valid identifier'.format(name))
    if mbn.is_a_number(lower_bounds) and mbn.is_a_number(upper_bounds) and (lower_bounds > upper_bounds):
        raise ValueError('lower_bound={} is greater than upper_bound={} for variable set={}'.format(lower_bounds, upper_bounds, name))
    if isinstance(is_integral, bool) and is_integral and mbn.is_a_number(lower_bounds) and mbn.is_a_number(upper_bounds) and math.isfinite(lower_bounds) and math.isfinite(upper_bounds) and (math.ceil(lower_bounds) > math.floor(upper_bounds)):
        raise ValueError('ceil(lower_bound={})={}'.format(lower_bounds, math.ceil(lower_bounds)) + ' is greater than floor(' + 'upper_bound={})={}'.format(upper_bounds, math.floor(upper_bounds)) + ' for variable set={}'.format(name))
    lower_bounds = _convert_to_series_and_validate_index(lower_bounds, index)
    upper_bounds = _convert_to_series_and_validate_index(upper_bounds, index)
    is_integrals = _convert_to_series_and_validate_index(is_integral, index)
    return pd.Series(index=index, data=[Variable(helper=self.__helper, name=f'{name}[{i}]', lb=lower_bounds[i], ub=upper_bounds[i], is_integral=is_integrals[i]) for i in index])