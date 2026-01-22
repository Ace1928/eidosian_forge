import collections
import dataclasses
import functools
import io
import itertools
import threading
from absl import app
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.util import nest
def topological_sort_operations(operations):
    """Topological sorts a list of operations.

  This does a topological sort of the operations in a graph. The edges include
  both data dependencies and control dependencies. Note that the edge goes from
  an operation to its dependencies.

  The sort is intentionally unstable, reversing orders of operations and
  dependencies on ties.

  Args:
    operations: a list of tf.Operation in the same graph.

  Returns:
    A map from a tf.Operation to its topological order.
  """
    in_degrees = collections.OrderedDict()
    for op in reversed(operations):
        if op not in in_degrees:
            in_degrees[op] = 0
        for next_op in reversed(_op_dependencies(op)):
            in_degrees[next_op] = in_degrees.get(next_op, 0) + 1
    nexts = []
    for op, in_degree in in_degrees.items():
        if in_degree == 0:
            nexts.append(op)
    order = {}
    next_order = 0
    while nexts:
        op, nexts = (nexts[0], nexts[1:])
        order[op] = next_order
        next_order += 1
        for next_op in reversed(_op_dependencies(op)):
            in_degrees[next_op] -= 1
            if in_degrees[next_op] == 0:
                nexts.append(next_op)
    assert len(order) == len(operations)
    return order