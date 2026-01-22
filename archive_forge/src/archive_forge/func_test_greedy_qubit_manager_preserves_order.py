import cirq
def test_greedy_qubit_manager_preserves_order():
    qm = cirq.GreedyQubitManager(prefix='anc')
    ancillae = [cirq.q(f'anc_{i}') for i in range(100)]
    assert qm.qalloc(100) == ancillae
    qm.qfree(ancillae)
    assert qm.qalloc(100) == ancillae