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
def new_var(self, lb: NumberT, ub: NumberT, is_integer: bool, name: Optional[str]) -> Variable:
    """Create an integer variable with domain [lb, ub].

        Args:
          lb: Lower bound of the variable.
          ub: Upper bound of the variable.
          is_integer: Indicates if the variable must take integral values.
          name: The name of the variable.

        Returns:
          a variable whose domain is [lb, ub].
        """
    return Variable(self.__helper, lb, ub, is_integer, name)