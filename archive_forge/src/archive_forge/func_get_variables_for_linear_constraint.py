from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def get_variables_for_linear_constraint(self, linear_constraint_id: int) -> Iterator[int]:
    self._check_linear_constraint_id(linear_constraint_id)
    yield from self.linear_constraints[linear_constraint_id].variable_nonzeros