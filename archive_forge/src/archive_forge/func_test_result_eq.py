import pytest
import pandas as pd
import sympy
import cirq
import cirq.experiments.t2_decay_experiment as t2
def test_result_eq():
    example_data = pd.DataFrame(columns=['delay_ns', 0, 1], index=range(5), data=[[200.0, 0, 100], [400.0, 20, 80], [600.0, 40, 60], [800.0, 60, 40], [1000.0, 80, 20]])
    other_data = pd.DataFrame(columns=['delay_ns', 0, 1], index=range(5), data=[[200.0, 0, 100], [400.0, 19, 81], [600.0, 39, 61], [800.0, 59, 41], [1000.0, 79, 21]])
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.experiments.T2DecayResult(example_data, example_data))
    eq.add_equality_group(cirq.experiments.T2DecayResult(other_data, example_data))
    eq.add_equality_group(cirq.experiments.T2DecayResult(example_data, other_data))