import itertools
import random
import pytest
import networkx
import cirq
def test_to_circuit():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.Y(q0))
    dag = cirq.contrib.CircuitDag.from_circuit(circuit)
    assert networkx.dag.is_directed_acyclic_graph(dag)
    assert circuit == dag.to_circuit()
    cirq.testing.assert_allclose_up_to_global_phase(circuit.unitary(), dag.to_circuit().unitary(), atol=1e-07)