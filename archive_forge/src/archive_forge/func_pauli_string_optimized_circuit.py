import networkx
from cirq import circuits, linalg
from cirq.contrib import circuitdag
from cirq.contrib.paulistring.pauli_string_dag import pauli_string_dag_from_circuit
from cirq.contrib.paulistring.recombine import move_pauli_strings_into_circuit
from cirq.contrib.paulistring.separate import convert_and_separate_circuit
from cirq.ops import PauliStringGateOperation
def pauli_string_optimized_circuit(circuit: circuits.Circuit, move_cliffords: bool=True, atol: float=1e-08) -> circuits.Circuit:
    cl, cr = convert_and_separate_circuit(circuit, leave_cliffords=not move_cliffords, atol=atol)
    string_dag = pauli_string_dag_from_circuit(cl)
    while True:
        before_len = len(string_dag.nodes())
        merge_equal_strings(string_dag)
        remove_negligible_strings(string_dag)
        if len(string_dag.nodes()) >= before_len:
            break
    c_all = move_pauli_strings_into_circuit(string_dag, cr)
    assert_no_multi_qubit_pauli_strings(c_all)
    return c_all