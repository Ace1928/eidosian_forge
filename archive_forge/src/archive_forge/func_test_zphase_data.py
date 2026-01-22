import cirq, cirq_google
from cirq.devices.noise_utils import OpIdentifier
from google.protobuf.text_format import Merge
import numpy as np
import pytest
def test_zphase_data():
    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)]
    pauli_error = [0.001, 0.002, 0.003]
    incoherent_error = [0.0001, 0.0002, 0.0003]
    p00_error = [0.004, 0.005, 0.006]
    p11_error = [0.007, 0.008, 0.009]
    t1_micros = [10, 20, 30]
    syc_pauli = [0.01, 0.02]
    iswap_pauli = [0.03, 0.04]
    syc_angles = [cirq.PhasedFSimGate(theta=0.011, phi=-0.021, zeta=-0.031, gamma=0.043), cirq.PhasedFSimGate(theta=-0.012, phi=0.022, zeta=0.032, gamma=-0.044)]
    iswap_angles = [cirq.PhasedFSimGate(theta=-0.013, phi=0.023, zeta=0.031, gamma=-0.043), cirq.PhasedFSimGate(theta=0.014, phi=-0.024, zeta=-0.032, gamma=0.044)]
    calibration = get_mock_calibration(pauli_error, incoherent_error, p00_error, p11_error, t1_micros, syc_pauli, iswap_pauli, syc_angles, iswap_angles)
    qubit_pairs = [(qubits[0], qubits[1]), (qubits[0], qubits[2])]
    zphase_data = {'syc': {'zeta': {qubit_pairs[0]: syc_angles[0].zeta, qubit_pairs[1]: syc_angles[1].zeta}, 'gamma': {qubit_pairs[0]: syc_angles[0].gamma, qubit_pairs[1]: syc_angles[1].gamma}}, 'sqrt_iswap': {'zeta': {qubit_pairs[0]: iswap_angles[0].zeta, qubit_pairs[1]: iswap_angles[1].zeta}, 'gamma': {qubit_pairs[0]: iswap_angles[0].gamma, qubit_pairs[1]: iswap_angles[1].gamma}}}
    prop = cirq_google.noise_properties_from_calibration(calibration, zphase_data)
    for i, qs in enumerate(qubit_pairs):
        for gate, values in [(cirq_google.SycamoreGate, syc_angles), (cirq.ISwapPowGate, iswap_angles)]:
            assert prop.fsim_errors[OpIdentifier(gate, *qs)] == values[i]
            assert prop.fsim_errors[OpIdentifier(gate, *qs[::-1])] == values[i]
            assert prop.fsim_errors[OpIdentifier(gate, *qs)] == values[i]
            assert prop.fsim_errors[OpIdentifier(gate, *qs[::-1])] == values[i]