import itertools
import random
import pytest
import networkx
import cirq
def test_from_circuit():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.Y(q0))
    dag = cirq.contrib.CircuitDag.from_circuit(circuit)
    assert networkx.dag.is_directed_acyclic_graph(dag)
    assert len(dag.nodes()) == 2
    assert [(n1.val, n2.val) for n1, n2 in dag.edges()] == [(cirq.X(q0), cirq.Y(q0))]
    assert sorted(circuit.all_qubits()) == sorted(dag.all_qubits())