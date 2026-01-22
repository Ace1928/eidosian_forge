from typing import cast, Sequence, TYPE_CHECKING
from cirq import devices, ops, protocols
from cirq.contrib.acquaintance.permutation import PermutationGate, update_mapping
def assert_permutation_decomposition_equivalence(gate: PermutationGate, n_qubits: int) -> None:
    qubits = devices.LineQubit.range(n_qubits)
    operations = protocols.decompose_once_with_qubits(gate, qubits)
    operations = list(cast(Sequence['cirq.Operation'], ops.flatten_op_tree(operations)))
    mapping = {cast(ops.Qid, q): i for i, q in enumerate(qubits)}
    update_mapping(mapping, operations)
    expected_mapping = {qubits[j]: i for i, j in gate.permutation().items()}
    assert mapping == expected_mapping, f"{gate!r}.permutation({n_qubits}) doesn't match decomposition.\n\nActual mapping:\n{[mapping[q] for q in qubits]}\n\nExpected mapping:\n{[expected_mapping[q] for q in qubits]}\n"