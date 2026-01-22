import functools
import itertools
import math
import operator
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, TYPE_CHECKING
from cirq import ops, protocols, value
from cirq.contrib.acquaintance.shift import CircularShiftGate
from cirq.contrib.acquaintance.permutation import (
@staticmethod
def from_operations(qubit_order: Sequence['cirq.Qid'], operations: Sequence['cirq.Operation'], acquaintance_size: Optional[int]=0, swap_gate: 'cirq.Gate'=ops.SWAP) -> 'SwapNetworkGate':
    part_sizes = operations_to_part_lens(qubit_order, operations)
    return SwapNetworkGate(part_sizes, acquaintance_size, swap_gate)