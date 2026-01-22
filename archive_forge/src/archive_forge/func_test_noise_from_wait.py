import numpy as np
import pytest
import sympy
import cirq
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
from cirq.devices.thermal_noise_model import (
def test_noise_from_wait():
    q0 = cirq.LineQubit(0)
    gate_durations = {cirq.ZPowGate: 25.0}
    heat_rate_GHz = {q0: 1e-05}
    cool_rate_GHz = {q0: 0.0001}
    model = ThermalNoiseModel(qubits={q0}, gate_durations_ns=gate_durations, heat_rate_GHz=heat_rate_GHz, cool_rate_GHz=cool_rate_GHz, dephase_rate_GHz=None, require_physical_tag=False, skip_measurements=True)
    moment = cirq.Moment(cirq.wait(q0, nanos=100))
    noisy_moment = model.noisy_moment(moment, system_qubits=[q0])
    assert noisy_moment[0] == moment
    assert len(noisy_moment[1]) == 1
    noisy_choi = cirq.kraus_to_choi(cirq.kraus(noisy_moment[1].operations[0]))
    assert np.allclose(noisy_choi, [[0.99900548, 0, 0, 0.994515097], [0, 0.00994520111, 0, 0], [0, 0, 0.000994520111, 0], [0.994515097, 0, 0, 0.990054799]])