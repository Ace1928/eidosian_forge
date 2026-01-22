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
def dual_values(self, constraints: _IndexOrSeries) -> pd.Series:
    """Returns the dual values of the input constraints.

        If `constraints` is a `pd.Index`, then the output will be indexed by the
        constraints. If `constraints` is a `pd.Series` indexed by the underlying
        dimensions, then the output will be indexed by the same underlying
        dimensions.

        Args:
          constraints (Union[pd.Index, pd.Series]): The set of constraints from
            which to get the dual values.

        Returns:
          pd.Series: The dual_values of all constraints in the set.
        """
    if not self.__solve_helper.has_solution():
        return _attribute_series(func=lambda v: pd.NA, values=constraints)
    return _attribute_series(func=lambda v: self.__solve_helper.dual_value(v.index), values=constraints)