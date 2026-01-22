import itertools
import pytest
import cirq
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('subgraph,part_size', itertools.product(cca.BipartiteGraphType, range(1, 6)))
def test_decomposition_permutation_consistency(part_size, subgraph):
    gate = cca.BipartiteSwapNetworkGate(subgraph, part_size)
    qubits = cirq.LineQubit.range(2 * part_size)
    mapping = {q: i for i, q in enumerate(qubits)}
    cca.update_mapping(mapping, gate._decompose_(qubits))
    permutation = gate.permutation()
    assert {qubits[i]: j for i, j in permutation.items()} == mapping