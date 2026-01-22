from typing import Iterable, List, Set, TYPE_CHECKING
from cirq.ops import named_qubit, qid_util, qubit_manager
def qfree(self, qubits: Iterable['cirq.Qid']) -> None:
    qs = list(dict(zip(qubits, qubits)).keys())
    assert self._used_qubits.issuperset(qs), 'Only managed qubits currently in-use can be freed'
    self._used_qubits = self._used_qubits.difference(qs)
    self._free_qubits.extend(qs)