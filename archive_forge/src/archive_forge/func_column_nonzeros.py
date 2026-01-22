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
def column_nonzeros(self, variable: Variable) -> Iterator[LinearConstraint]:
    """Yields the linear constraints with nonzero coefficient for this variable."""
    for lin_con_id in self.storage.get_linear_constraints_with_variable(variable.id):
        yield self._get_or_make_linear_constraint(lin_con_id)