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
def get_linear_constraint(self, lin_con_id: int) -> LinearConstraint:
    """Returns the LinearConstraint for the id lin_con_id."""
    if not self.storage.linear_constraint_exists(lin_con_id):
        raise KeyError(f'linear constraint does not exist with id {lin_con_id}')
    return self._get_or_make_linear_constraint(lin_con_id)