import itertools
import random
import pytest
import networkx
import cirq
def test_to_empty_circuit():
    circuit = cirq.Circuit()
    dag = cirq.contrib.CircuitDag.from_circuit(circuit)
    assert networkx.dag.is_directed_acyclic_graph(dag)
    assert circuit == dag.to_circuit()