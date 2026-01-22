import abc
import itertools
from typing import Iterable, Optional, TYPE_CHECKING, Tuple, cast
from cirq import devices, ops, value
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
def duration_of(self, operation: ops.Operation) -> value.Duration:
    return self.get_device_edge_from_op(operation).duration_of(operation)