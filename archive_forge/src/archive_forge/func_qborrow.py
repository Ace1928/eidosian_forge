from typing import Iterable, List, Set, TYPE_CHECKING
from cirq.ops import named_qubit, qid_util, qubit_manager
def qborrow(self, n: int, dim: int=2) -> List['cirq.Qid']:
    return self.qalloc(n, dim)