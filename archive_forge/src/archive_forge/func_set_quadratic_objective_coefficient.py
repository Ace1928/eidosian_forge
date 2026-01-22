from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def set_quadratic_objective_coefficient(self, first_variable_id: int, second_variable_id: int, value: float) -> None:
    self._check_variable_id(first_variable_id)
    self._check_variable_id(second_variable_id)
    updated = self._quadratic_objective_coefficients.set_coefficient(first_variable_id, second_variable_id, value)
    if updated:
        for watcher in self._update_trackers:
            if max(first_variable_id, second_variable_id) < watcher.variables_checkpoint:
                watcher.quadratic_objective_coefficients.add(_QuadraticKey(first_variable_id, second_variable_id))