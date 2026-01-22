from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def get_adjacent_variables(self, variable_id: int) -> Iterator[int]:
    """Yields the variables multiplying a variable in the stored quadratic terms.

        If variable_id is not in the model the function yields the empty set.

        Args:
          variable_id: Function yields the variables multiplying variable_id in the
            stored quadratic terms.

        Yields:
          The variables multiplying variable_id in the stored quadratic terms.
        """
    yield from self._adjacent_variables.get(variable_id, ())