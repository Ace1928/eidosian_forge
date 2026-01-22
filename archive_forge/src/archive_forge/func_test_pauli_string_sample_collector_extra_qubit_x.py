import duet
import cirq
def test_pauli_string_sample_collector_extra_qubit_x():
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliSumCollector(circuit=cirq.Circuit(cirq.H(a)), observable=3 * cirq.X(b), samples_per_term=10000)
    p.collect(sampler=cirq.Simulator())
    assert abs(p.estimated_energy()) < 0.5