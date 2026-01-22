from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
def test_sampler_sample_multiple_params():
    a, b = cirq.LineQubit.range(2)
    s = sympy.Symbol('s')
    t = sympy.Symbol('t')
    sampler = cirq.Simulator()
    circuit = cirq.Circuit(cirq.X(a) ** s, cirq.X(b) ** t, cirq.measure(a, b, key='out'))
    results = sampler.sample(circuit, repetitions=3, params=[{'s': 0, 't': 0}, {'s': 0, 't': 1}, {'s': 1, 't': 0}, {'s': 1, 't': 1}])
    pd.testing.assert_frame_equal(results, pd.DataFrame(columns=['s', 't', 'out'], index=[0, 1, 2] * 4, data=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [1, 0, 2], [1, 0, 2], [1, 0, 2], [1, 1, 3], [1, 1, 3], [1, 1, 3]]))