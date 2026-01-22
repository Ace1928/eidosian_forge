import pytest
import pandas as pd
import sympy
import cirq
import cirq.experiments.t2_decay_experiment as t2
def test_cpmg_sweep():
    sweep = t2._cpmg_sweep([1, 3, 5])
    expected = cirq.Zip(cirq.Points('pulse_0', [1, 1, 1]), cirq.Points('pulse_1', [0, 1, 1]), cirq.Points('pulse_2', [0, 1, 1]), cirq.Points('pulse_3', [0, 0, 1]), cirq.Points('pulse_4', [0, 0, 1]))
    assert sweep == expected