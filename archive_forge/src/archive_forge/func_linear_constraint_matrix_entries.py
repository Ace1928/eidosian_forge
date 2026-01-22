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
def linear_constraint_matrix_entries(self) -> Iterator[LinearConstraintMatrixEntry]:
    """Yields the nonzero elements of the linear constraint matrix in undefined order."""
    for entry in self.storage.get_linear_constraint_matrix_entries():
        yield LinearConstraintMatrixEntry(linear_constraint=self._get_or_make_linear_constraint(entry.linear_constraint_id), variable=self._get_or_make_variable(entry.variable_id), coefficient=entry.coefficient)