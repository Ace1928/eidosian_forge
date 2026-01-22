from typing import Iterable, List, Set, TYPE_CHECKING
from cirq.ops import named_qubit, qid_util, qubit_manager
def qalloc(self, n: int, dim: int=2) -> List['cirq.Qid']:
    if not n:
        return []
    self.resize(self._size + n - len(self._free_qubits), dim=dim)
    ret_qubits = self._free_qubits[-n:] if self.maximize_reuse else self._free_qubits[:n]
    self._free_qubits = self._free_qubits[:-n] if self.maximize_reuse else self._free_qubits[n:]
    self._used_qubits.update(ret_qubits)
    return ret_qubits