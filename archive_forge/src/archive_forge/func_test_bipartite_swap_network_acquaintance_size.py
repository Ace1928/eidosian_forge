import itertools
import pytest
import cirq
import cirq.contrib.acquaintance as cca
def test_bipartite_swap_network_acquaintance_size():
    qubits = cirq.LineQubit.range(4)
    gate = cca.BipartiteSwapNetworkGate(cca.BipartiteGraphType.COMPLETE, 2)
    assert cca.get_acquaintance_size(gate(*qubits)) == 2