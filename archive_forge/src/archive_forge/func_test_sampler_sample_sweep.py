from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
def test_sampler_sample_sweep():
    a = cirq.LineQubit(0)
    t = sympy.Symbol('t')
    sampler = cirq.Simulator()
    circuit = cirq.Circuit(cirq.X(a) ** t, cirq.measure(a, key='out'))
    results = sampler.sample(circuit, repetitions=3, params=cirq.Linspace('t', 0, 2, 3))
    pd.testing.assert_frame_equal(results, pd.DataFrame(columns=['t', 'out'], index=[0, 1, 2] * 3, data=[[0.0, 0], [0.0, 0], [0.0, 0], [1.0, 1], [1.0, 1], [1.0, 1], [2.0, 0], [2.0, 0], [2.0, 0]]))