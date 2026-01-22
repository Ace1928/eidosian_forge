import itertools
import random
import pytest
import networkx
import cirq
def test_larger_circuit():
    q0, q1, q2, q3 = [cirq.GridQubit(0, 5), cirq.GridQubit(1, 5), cirq.GridQubit(2, 5), cirq.GridQubit(3, 5)]
    circuit = cirq.Circuit(cirq.X(q0), cirq.CZ(q1, q2), cirq.CZ(q0, q1), cirq.Y(q0), cirq.Z(q0), cirq.CZ(q1, q2), cirq.X(q0), cirq.Y(q0), cirq.CZ(q0, q1), cirq.T(q3), strategy=cirq.InsertStrategy.EARLIEST)
    dag = cirq.contrib.CircuitDag.from_circuit(circuit)
    assert networkx.dag.is_directed_acyclic_graph(dag)
    desired = '\n(0, 5): ───X───@───Y───Z───X───Y───@───\n               │                   │\n(1, 5): ───@───@───@───────────────@───\n           │       │\n(2, 5): ───@───────@───────────────────\n\n(3, 5): ───T───────────────────────────\n'
    cirq.testing.assert_has_diagram(circuit, desired)
    cirq.testing.assert_has_diagram(dag.to_circuit(), desired)
    cirq.testing.assert_allclose_up_to_global_phase(circuit.unitary(), dag.to_circuit().unitary(), atol=1e-07)