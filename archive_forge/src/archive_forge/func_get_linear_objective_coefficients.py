from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def get_linear_objective_coefficients(self) -> Iterator[model_storage.LinearObjectiveEntry]:
    for var_id, coef in self.linear_objective_coefficient.items():
        yield model_storage.LinearObjectiveEntry(variable_id=var_id, coefficient=coef)