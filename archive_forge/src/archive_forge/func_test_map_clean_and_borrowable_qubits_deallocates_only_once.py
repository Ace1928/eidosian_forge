import cirq
def test_map_clean_and_borrowable_qubits_deallocates_only_once():
    q = [cirq.ops.BorrowableQubit(i) for i in range(2)] + [cirq.q('q')]
    circuit = cirq.Circuit(cirq.X.on_each(*q), cirq.Y(q[1]), cirq.Z(q[1]))
    greedy_mm = cirq.GreedyQubitManager(prefix='a', size=2)
    mapped_circuit = cirq.map_clean_and_borrowable_qubits(circuit, qm=greedy_mm)
    cirq.testing.assert_has_diagram(mapped_circuit, '\na_0: ───X───────────\n\na_1: ───X───Y───Z───\n\nq: ─────X───────────\n')