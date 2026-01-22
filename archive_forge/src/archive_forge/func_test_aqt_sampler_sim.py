from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
from cirq_aqt import AQTSampler, AQTSamplerLocalSimulator
from cirq_aqt.aqt_device import get_aqt_device, get_op_string
def test_aqt_sampler_sim():
    theta = sympy.Symbol('theta')
    num_points = 10
    max_angle = np.pi
    repetitions = 1000
    num_qubits = 4
    _, qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerLocalSimulator()
    sampler.simulate_ideal = True
    circuit = cirq.Circuit(cirq.X(qubits[3]) ** theta, cirq.X(qubits[0]), cirq.X(qubits[0]), cirq.X(qubits[1]), cirq.X(qubits[1]), cirq.X(qubits[2]), cirq.X(qubits[2]))
    circuit.append(cirq.PhasedXPowGate(phase_exponent=0.5, exponent=-0.5).on(qubits[0]))
    circuit.append(cirq.PhasedXPowGate(phase_exponent=0.5, exponent=0.5).on(qubits[0]))
    sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
    results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)
    excited_state_probs = np.zeros(num_points)
    for i in range(num_points):
        excited_state_probs[i] = np.mean(results[i].measurements['m'])
    assert excited_state_probs[-1] == 0.25