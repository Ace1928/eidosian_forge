from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
def test_sampler_sample_expectation_values_multi_param():
    a = cirq.LineQubit(0)
    t = sympy.Symbol('t')
    sampler = cirq.Simulator(seed=1)
    circuit = cirq.Circuit(cirq.X(a) ** t)
    obs = cirq.Z(a)
    results = sampler.sample_expectation_values(circuit, [obs], num_samples=5, params=cirq.Linspace('t', 0, 2, 3))
    assert np.allclose(results, [[1], [-1], [1]])