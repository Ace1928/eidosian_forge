import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_sample_sweep():
    q = cirq.NamedQubit('q')
    c = cirq.Circuit(cirq.X(q), cirq.Y(q) ** sympy.Symbol('t'), cirq.measure(q))
    results = cirq.sample_sweep(c, cirq.Linspace('t', 0, 1, 2), repetitions=3)
    assert len(results) == 2
    assert results[0].histogram(key=q) == collections.Counter({1: 3})
    assert results[1].histogram(key=q) == collections.Counter({0: 3})
    c = cirq.Circuit(cirq.X(q), cirq.amplitude_damp(1).on(q), cirq.Y(q) ** sympy.Symbol('t'), cirq.measure(q))
    results = cirq.sample_sweep(c, cirq.Linspace('t', 0, 1, 2), repetitions=3)
    assert len(results) == 2
    assert results[0].histogram(key=q) == collections.Counter({0: 3})
    assert results[1].histogram(key=q) == collections.Counter({1: 3})
    c = cirq.Circuit(cirq.X(q), cirq.Y(q) ** sympy.Symbol('t'), cirq.measure(q))
    results = cirq.sample_sweep(c, cirq.Linspace('t', 0, 1, 2), noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(1)), repetitions=3)
    assert len(results) == 2
    assert results[0].histogram(key=q) == collections.Counter({0: 3})
    assert results[1].histogram(key=q) == collections.Counter({0: 3})