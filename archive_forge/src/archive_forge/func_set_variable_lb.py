from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def set_variable_lb(self, variable_id: int, lb: float) -> None:
    self._check_variable_id(variable_id)
    if lb == self.variables[variable_id].lower_bound:
        return
    self.variables[variable_id].lower_bound = lb
    for watcher in self._update_trackers:
        if variable_id < watcher.variables_checkpoint:
            watcher.variable_lbs.add(variable_id)