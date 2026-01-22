from itertools import product, combinations
import pytest
import cirq
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('n_qubits, acquaintance_size', product(range(2, 6), range(2, 5)))
def test_get_logical_acquaintance_opportunities(n_qubits, acquaintance_size):
    qubits = cirq.LineQubit.range(n_qubits)
    acquaintance_strategy = cca.complete_acquaintance_strategy(qubits, acquaintance_size)
    initial_mapping = {q: i for i, q in enumerate(qubits)}
    opps = cca.get_logical_acquaintance_opportunities(acquaintance_strategy, initial_mapping)
    assert opps == set((frozenset(s) for s in combinations(range(n_qubits), acquaintance_size)))