import collections
import itertools
import numbers
import threading
import time
from typing import (
import warnings
import pandas as pd
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model_helper as cmh
from ortools.sat.python import swig_helper
from ortools.util.python import sorted_interval_list
def new_optional_fixed_size_interval_var(self, start: LinearExprT, size: IntegralT, is_present: LiteralT, name: str) -> IntervalVar:
    """Creates an interval variable from start, and a fixed size.

        An interval variable is a constraint, that is itself used in other
        constraints like NoOverlap.

        Args:
          start: The start of the interval. It must be of the form a * var + b.
          size: The size of the interval. It must be an integer value.
          is_present: A literal that indicates if the interval is active or not. A
            inactive interval is simply ignored by all constraints.
          name: The name of the interval variable.

        Returns:
          An `IntervalVar` object.
        """
    size = cmh.assert_is_int64(size)
    start_expr = self.parse_linear_expression(start)
    size_expr = self.parse_linear_expression(size)
    end_expr = self.parse_linear_expression(start + size)
    if len(start_expr.vars) > 1:
        raise TypeError('cp_model.new_interval_var: start must be affine or constant.')
    is_present_index = self.get_or_make_boolean_index(is_present)
    return IntervalVar(self.__model, start_expr, size_expr, end_expr, is_present_index, name)