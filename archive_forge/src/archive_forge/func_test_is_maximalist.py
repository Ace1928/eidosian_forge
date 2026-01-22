import itertools
import random
import pytest
import networkx
import cirq
@pytest.mark.parametrize('circuit', [cirq.testing.random_circuit(10, 10, 0.5) for _ in range(3)])
def test_is_maximalist(circuit):
    dag = cirq.contrib.CircuitDag.from_circuit(circuit)
    transitive_closure = networkx.dag.transitive_closure(dag)
    assert cirq.contrib.CircuitDag(incoming_graph_data=transitive_closure) == dag
    assert not any((dag.has_edge(b, a) for a, b in itertools.combinations(dag.ordered_nodes(), 2)))