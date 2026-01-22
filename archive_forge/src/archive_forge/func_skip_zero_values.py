from typing import Dict, FrozenSet, Generic, Iterable, Mapping, Optional, Set, TypeVar
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model
@property
def skip_zero_values(self) -> bool:
    return self._skip_zero_values