from typing import Dict, List, Tuple
from cirq.ops.fsim_gate import PhasedFSimGate
import numpy as np
import pytest
import cirq, cirq_google
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
from cirq_google.devices.google_noise_properties import (
@pytest.mark.parametrize('op', [cirq.ISWAP(*cirq.LineQubit.range(2)) ** 0.6, cirq.CZ(*cirq.LineQubit.range(2)) ** 0.3, cirq_google.SYC(*cirq.LineQubit.range(2))])
def test_two_qubit_gates(op):
    q0, q1 = cirq.LineQubit.range(2)
    props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    model = NoiseModelFromGoogleNoiseProperties(props)
    circuit = cirq.Circuit(op)
    noisy_circuit = circuit.with_noise(model)
    assert len(noisy_circuit.moments) == 4
    assert len(noisy_circuit.moments[0].operations) == 1
    assert noisy_circuit.moments[0].operations[0] == op.with_tags(PHYSICAL_GATE_TAG)
    assert len(noisy_circuit.moments[1].operations) == 1
    depol_op = noisy_circuit.moments[1].operations[0]
    assert isinstance(depol_op.gate, cirq.DepolarizingChannel)
    assert np.isclose(depol_op.gate.p, 0.00719705)
    assert len(noisy_circuit.moments[2].operations) == 1
    fsim_op = noisy_circuit.moments[2].operations[0]
    assert isinstance(fsim_op.gate, cirq.PhasedFSimGate)
    assert fsim_op == PhasedFSimGate(theta=0.01, zeta=0.03, chi=0.04, gamma=0.05, phi=0.02).on(q0, q1)
    assert len(noisy_circuit.moments[3].operations) == 2
    thermal_op_0 = noisy_circuit.moments[3].operation_at(q0)
    thermal_op_1 = noisy_circuit.moments[3].operation_at(q1)
    assert isinstance(thermal_op_0.gate, cirq.KrausChannel)
    assert isinstance(thermal_op_1.gate, cirq.KrausChannel)
    thermal_choi_0 = cirq.kraus_to_choi(cirq.kraus(thermal_op_0))
    thermal_choi_1 = cirq.kraus_to_choi(cirq.kraus(thermal_op_1))
    expected_thermal_choi = np.array([[1, 0, 0, 0.999680051], [0, 0.000319948805, 0, 0], [0, 0, 0, 0], [0.999680051, 0, 0, 0.999680051]])
    assert np.allclose(thermal_choi_0, expected_thermal_choi)
    assert np.allclose(thermal_choi_1, expected_thermal_choi)
    depol_pauli_err = 1 - cirq.qis.measures.entanglement_fidelity(depol_op)
    fsim_pauli_err = 1 - cirq.qis.measures.entanglement_fidelity(fsim_op)
    thermal0_pauli_err = 1 - cirq.qis.measures.entanglement_fidelity(thermal_op_0)
    thermal1_pauli_err = 1 - cirq.qis.measures.entanglement_fidelity(thermal_op_1)
    total_err = depol_pauli_err + thermal0_pauli_err + thermal1_pauli_err + fsim_pauli_err
    assert np.isclose(total_err, TWO_QUBIT_ERROR)