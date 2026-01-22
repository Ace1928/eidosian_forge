import abc
from typing import (
from cirq import circuits, ops, protocols, transformers, value
from cirq.type_workarounds import NotImplementedType
def return_to_initial_mapping(circuit: 'cirq.Circuit', swap_gate: 'cirq.Gate'=ops.SWAP) -> None:
    qubits = sorted(circuit.all_qubits())
    n_qubits = len(qubits)
    mapping = {q: i for i, q in enumerate(qubits)}
    update_mapping(mapping, circuit.all_operations())
    permutation = {i: mapping[q] for i, q in enumerate(qubits)}
    returning_permutation_op = LinearPermutationGate(n_qubits, permutation, swap_gate)(*qubits)
    circuit.append(returning_permutation_op)