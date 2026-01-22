from typing import List
import pytest
import sympy
import numpy as np
import cirq
import cirq_google as cg
def test_sweeps_validation():
    sampler = cg.ValidatingSampler(device=cirq.UNCONSTRAINED_DEVICE, validator=_too_many_reps, sampler=cirq.Simulator())
    q = cirq.GridQubit(2, 2)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m'))
    sweeps = [cirq.Points(key='t', points=[1, 0]), cirq.Points(key='x', points=[0, 1])]
    with pytest.raises(ValueError, match='Too many repetitions'):
        _ = sampler.run_sweep(circuit, sweeps, repetitions=20000)