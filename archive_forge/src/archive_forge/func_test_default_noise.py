import pytest
import cirq
from cirq_pasqal import PasqalNoiseModel, PasqalDevice
from cirq.ops import NamedQubit
def test_default_noise():
    p_qubits = cirq.NamedQubit.range(2, prefix='q')
    p_device = PasqalDevice(qubits=p_qubits)
    noise_model = PasqalNoiseModel(p_device)
    circuit = cirq.Circuit()
    Gate_l = cirq.ops.CZPowGate(exponent=2)
    circuit.append(Gate_l.on(p_qubits[0], p_qubits[1]))
    p_circuit = cirq.Circuit(circuit)
    n_mts = []
    for moment in p_circuit._moments:
        n_mts.append(noise_model.noisy_moment(moment, p_qubits))
    assert n_mts == [[cirq.ops.CZPowGate(exponent=2).on(NamedQubit('q0'), NamedQubit('q1')), cirq.depolarize(p=0.05).on(NamedQubit('q0')), cirq.depolarize(p=0.05).on(NamedQubit('q1'))]]