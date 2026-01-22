import networkx
from cirq import circuits, linalg
from cirq.contrib import circuitdag
from cirq.contrib.paulistring.pauli_string_dag import pauli_string_dag_from_circuit
from cirq.contrib.paulistring.recombine import move_pauli_strings_into_circuit
from cirq.contrib.paulistring.separate import convert_and_separate_circuit
from cirq.ops import PauliStringGateOperation
def assert_no_multi_qubit_pauli_strings(circuit: circuits.Circuit) -> None:
    for op in circuit.all_operations():
        if isinstance(op, PauliStringGateOperation):
            assert len(op.pauli_string) == 1, 'Multi qubit Pauli string left over'