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
def set_coefficient(self, var: Variable, coeff: NumberT) -> None:
    """Sets the coefficient of the variable in the constraint."""
    if self.is_always_false():
        raise ValueError(f'Constraint {self.index} is always false and cannot be modified')
    self.__helper.set_enforced_constraint_coefficient(self.__index, var.index, coeff)