import cirq
def make_circuit(qm: cirq.QubitManager):
    q = cirq.LineQubit.range(2)
    g = GateAllocInDecompose(1)
    context = cirq.DecompositionContext(qubit_manager=qm)
    circuit = cirq.Circuit(cirq.decompose_once(g.on(q[0]), context=context), cirq.decompose_once(g.on(q[1]), context=context))
    return circuit