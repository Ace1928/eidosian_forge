import itertools
import random
import pytest
import networkx
import cirq
@pytest.mark.parametrize('circuit, is_blocker', _get_circuits_and_is_blockers())
def test_findall_nodes_until_blocked(circuit, is_blocker):
    dag = cirq.contrib.CircuitDag.from_circuit(circuit)
    all_nodes = list(dag.ordered_nodes())
    found_nodes = list(dag.findall_nodes_until_blocked(is_blocker))
    assert not any((dag.has_edge(b, a) for a, b in itertools.combinations(found_nodes, 2)))
    blocking_nodes = set((node for node in all_nodes if is_blocker(node.val)))
    blocked_nodes = blocking_nodes.union(*(dag.succ[node] for node in blocking_nodes))
    expected_nodes = set(all_nodes) - blocked_nodes
    assert sorted(found_nodes) == sorted(expected_nodes)