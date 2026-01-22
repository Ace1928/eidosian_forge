import cirq
def test_greedy_qubit_manager():

    def make_circuit(qm: cirq.QubitManager):
        q = cirq.LineQubit.range(2)
        g = GateAllocInDecompose(1)
        context = cirq.DecompositionContext(qubit_manager=qm)
        circuit = cirq.Circuit(cirq.decompose_once(g.on(q[0]), context=context), cirq.decompose_once(g.on(q[1]), context=context))
        return circuit
    qm = cirq.GreedyQubitManager(prefix='ancilla', size=1)
    circuit = make_circuit(qm)
    cirq.testing.assert_has_diagram(circuit, '\n0: ───────────@───────\n              │\n1: ───────────┼───@───\n              │   │\nancilla_0: ───X───X───\n        ')
    qm = cirq.GreedyQubitManager(prefix='ancilla', size=2)
    circuit = make_circuit(qm)
    cirq.testing.assert_has_diagram(circuit, '\n              ┌──┐\n0: ────────────@─────\n               │\n1: ────────────┼@────\n               ││\nancilla_0: ────X┼────\n                │\nancilla_1: ─────X────\n              └──┘\n        ')
    qm = cirq.GreedyQubitManager(prefix='ancilla', size=2, maximize_reuse=True)
    circuit = make_circuit(qm)
    cirq.testing.assert_has_diagram(circuit, '\n0: ───────────@───────\n              │\n1: ───────────┼───@───\n              │   │\nancilla_1: ───X───X───\n     ')