import abc
import itertools
from typing import Iterable, Optional, TYPE_CHECKING, Tuple, cast
from cirq import devices, ops, value
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
def raise_crosstalk_error(*ops: ops.Operation):
    raise ValueError(f'crosstalk on {ops}')