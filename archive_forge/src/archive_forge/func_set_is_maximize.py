from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def set_is_maximize(self, is_maximize: bool) -> None:
    if self._is_maximize == is_maximize:
        return
    self._is_maximize = is_maximize
    for watcher in self._update_trackers:
        watcher.objective_direction = True