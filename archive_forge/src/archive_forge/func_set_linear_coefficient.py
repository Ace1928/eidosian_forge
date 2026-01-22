import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
def set_linear_coefficient(self, variable: Variable, coef: float) -> None:
    self.model.check_compatible(variable)
    self.model.storage.set_linear_objective_coefficient(variable.id, coef)