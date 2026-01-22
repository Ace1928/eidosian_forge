from typing import Any, Callable, Dict, Generic, Iterator, TypeVar, cast, TYPE_CHECKING
import functools
import networkx
from cirq import ops
from cirq.circuits import circuit
@staticmethod
def from_circuit(circuit: circuit.Circuit, can_reorder: Callable[['cirq.Operation', 'cirq.Operation'], bool]=_disjoint_qubits) -> 'CircuitDag':
    return CircuitDag.from_ops(circuit.all_operations(), can_reorder=can_reorder)