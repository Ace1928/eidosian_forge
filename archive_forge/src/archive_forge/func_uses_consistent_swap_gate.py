import abc
from typing import (
from cirq import circuits, ops, protocols, transformers, value
from cirq.type_workarounds import NotImplementedType
def uses_consistent_swap_gate(circuit: 'cirq.Circuit', swap_gate: 'cirq.Gate') -> bool:
    for op in circuit.all_operations():
        if isinstance(op, ops.GateOperation) and isinstance(op.gate, PermutationGate):
            if op.gate.swap_gate != swap_gate:
                return False
    return True