import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_sample_sweep_seed():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q))
    results = cirq.sample_sweep(circuit, [cirq.ParamResolver({'t': 0.5})] * 3, repetitions=2, seed=1234)
    assert np.all(results[0].measurements['q'] == [[False], [True]])
    assert np.all(results[1].measurements['q'] == [[False], [True]])
    assert np.all(results[2].measurements['q'] == [[True], [False]])
    results = cirq.sample_sweep(circuit, [cirq.ParamResolver({'t': 0.5})] * 3, repetitions=2, seed=np.random.RandomState(1234))
    assert np.all(results[0].measurements['q'] == [[False], [True]])
    assert np.all(results[1].measurements['q'] == [[False], [True]])
    assert np.all(results[2].measurements['q'] == [[True], [False]])