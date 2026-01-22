import pytest
import cirq
from cirq_aqt import AQTSimulator
from cirq_aqt.aqt_device import get_aqt_device
from cirq_aqt.aqt_device import AQTNoiseModel
def test_x_crosstalk_n_noise():
    num_qubits = 4
    noise_mod = AQTNoiseModel()
    _, qubits = get_aqt_device(num_qubits)
    circuit = cirq.Circuit()
    circuit.append(cirq.Y(qubits[1]) ** 0.5)
    circuit.append(cirq.Z(qubits[1]) ** 0.5)
    circuit.append(cirq.X(qubits[1]) ** 0.5)
    for moment in circuit.moments:
        noisy_moment = noise_mod.noisy_moment(moment, qubits)
    assert noisy_moment == [(cirq.X ** 0.5).on(cirq.LineQubit(1)), cirq.depolarize(p=0.001).on(cirq.LineQubit(1)), (cirq.X ** 0.015).on(cirq.LineQubit(0)), (cirq.X ** 0.015).on(cirq.LineQubit(2))]