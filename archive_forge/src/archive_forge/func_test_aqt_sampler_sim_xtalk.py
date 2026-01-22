from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
from cirq_aqt import AQTSampler, AQTSamplerLocalSimulator
from cirq_aqt.aqt_device import get_aqt_device, get_op_string
def test_aqt_sampler_sim_xtalk():
    num_points = 10
    max_angle = np.pi
    repetitions = 100
    num_qubits = 4
    _, qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerLocalSimulator()
    sampler.simulate_ideal = False
    circuit = cirq.Circuit(cirq.X(qubits[0]), cirq.X(qubits[1]), cirq.X(qubits[1]), cirq.X(qubits[3]), cirq.X(qubits[2]))
    sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
    _results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)