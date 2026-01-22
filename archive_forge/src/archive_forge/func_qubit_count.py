import functools
import itertools
from typing import Dict, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING
from cirq import ops
from cirq.contrib.acquaintance.gates import acquaint
from cirq.contrib.acquaintance.permutation import PermutationGate
from cirq.contrib.acquaintance.shift import CircularShiftGate
def qubit_count(self, side: Optional[str]=None) -> int:
    if side is None:
        return sum((self.qubit_count(side) for side in self.part_lens))
    return sum(self.part_lens[side])