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
def reduced_costs(self, variables: _IndexOrSeries) -> pd.Series:
    """Returns the reduced cost of the input variables.

        If `variables` is a `pd.Index`, then the output will be indexed by the
        variables. If `variables` is a `pd.Series` indexed by the underlying
        dimensions, then the output will be indexed by the same underlying
        dimensions.

        Args:
          variables (Union[pd.Index, pd.Series]): The set of variables from which to
            get the values.

        Returns:
          pd.Series: The reduced cost of all variables in the set.
        """
    if not self.__solve_helper.has_solution():
        return _attribute_series(func=lambda v: pd.NA, values=variables)
    return _attribute_series(func=lambda v: self.__solve_helper.reduced_cost(v.index), values=variables)