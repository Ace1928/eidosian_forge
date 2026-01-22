from typing import cast
from cirq import circuits, ops, protocols
from cirq.contrib import circuitdag
def pauli_string_reorder_pred(op1: ops.Operation, op2: ops.Operation) -> bool:
    ps1 = cast(ops.PauliStringGateOperation, op1).pauli_string
    ps2 = cast(ops.PauliStringGateOperation, op2).pauli_string
    return protocols.commutes(ps1, ps2)