import numpy as np
import pytest
import sympy
import cirq
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
from cirq.devices.thermal_noise_model import (
def test_noisy_moment_one_qubit():
    q0, q1 = cirq.LineQubit.range(2)
    model = ThermalNoiseModel(qubits={q0, q1}, gate_durations_ns={cirq.PhasedXZGate: 25.0, cirq.CZPowGate: 25.0}, heat_rate_GHz={q0: 1e-05, q1: 2e-05}, cool_rate_GHz={q0: 0.0001, q1: 0.0002}, dephase_rate_GHz={q0: 0.0003, q1: 0.0004}, require_physical_tag=False)
    gate = cirq.PhasedXZGate(x_exponent=1, z_exponent=0.5, axis_phase_exponent=0.25)
    moment = cirq.Moment(gate.on(q0))
    noisy_moment = model.noisy_moment(moment, system_qubits=[q0, q1])
    assert noisy_moment[0] == moment
    assert len(noisy_moment[1]) == 2
    noisy_choi = cirq.kraus_to_choi(cirq.kraus(noisy_moment[1].operations[0]))
    assert np.allclose(noisy_choi, [[0.999750343, 0, 0, 0.991164267], [0, 0.00249656565, 0, 0], [0, 0, 0.000249656565, 0], [0.991164267, 0, 0, 0.997503434]])