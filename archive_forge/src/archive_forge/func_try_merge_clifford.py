from typing import Tuple, cast
from cirq import circuits, ops, protocols, transformers
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def try_merge_clifford(cliff_op: ops.GateOperation, start_i: int) -> bool:
    orig_qubit, = cliff_op.qubits
    remaining_cliff_gate = ops.SingleQubitCliffordGate.I
    for pauli, quarter_turns in reversed(cast(ops.SingleQubitCliffordGate, cliff_op.gate).decompose_rotation()):
        trans = remaining_cliff_gate.pauli_tuple(pauli)
        pauli = trans[0]
        quarter_turns *= -1 if trans[1] else 1
        string_op = ops.PauliStringPhasor(ops.PauliString(pauli(cliff_op.qubits[0])), exponent_neg=quarter_turns / 2)
        merge_i, merge_op, num_passed = find_merge_point(start_i, string_op, quarter_turns == 2)
        assert merge_i > start_i
        assert len(merge_op.pauli_string) == 1, 'PauliString length != 1'
        assert not protocols.is_parameterized(merge_op.pauli_string)
        coefficient = merge_op.pauli_string.coefficient
        assert isinstance(coefficient, complex)
        qubit, pauli = next(iter(merge_op.pauli_string.items()))
        quarter_turns = round(merge_op.exponent_relative * 2)
        quarter_turns *= int(coefficient.real)
        quarter_turns %= 4
        part_cliff_gate = ops.SingleQubitCliffordGate.from_quarter_turns(pauli, quarter_turns)
        other_op = all_ops[merge_i] if merge_i < len(all_ops) else None
        if other_op is not None and qubit not in set(other_op.qubits):
            other_op = None
        if isinstance(other_op, ops.GateOperation) and isinstance(other_op.gate, ops.SingleQubitCliffordGate):
            new_op = part_cliff_gate.merged_with(other_op.gate)(qubit)
            all_ops[merge_i] = new_op
        elif isinstance(other_op, ops.GateOperation) and isinstance(other_op.gate, ops.CZPowGate) and (other_op.gate.exponent == 1) and (quarter_turns == 2):
            if pauli != ops.pauli_gates.Z:
                other_qubit = other_op.qubits[other_op.qubits.index(qubit) - 1]
                all_ops.insert(merge_i + 1, ops.SingleQubitCliffordGate.Z(other_qubit))
            all_ops.insert(merge_i + 1, part_cliff_gate(qubit))
        elif isinstance(other_op, ops.PauliStringPhasor):
            mod_op = other_op.pass_operations_over([part_cliff_gate(qubit)])
            all_ops[merge_i] = mod_op
            all_ops.insert(merge_i + 1, part_cliff_gate(qubit))
        elif merge_i > start_i + 1 and num_passed > 0:
            all_ops.insert(merge_i, part_cliff_gate(qubit))
        else:
            remaining_cliff_gate = remaining_cliff_gate.merged_with(part_cliff_gate)
    if remaining_cliff_gate == ops.SingleQubitCliffordGate.I:
        all_ops.pop(start_i)
        return True
    all_ops[start_i] = remaining_cliff_gate(orig_qubit)
    return False