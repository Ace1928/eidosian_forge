import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_result_addition_valid():
    a = cirq.ResultDict(params=cirq.ParamResolver({'ax': 1}), measurements={'q0': np.array([[0, 1], [1, 0], [0, 1]], dtype=bool), 'q1': np.array([[0], [0], [1]], dtype=bool)})
    b = cirq.ResultDict(params=cirq.ParamResolver({'ax': 1}), measurements={'q0': np.array([[0, 1]], dtype=bool), 'q1': np.array([[0]], dtype=bool)})
    c = a + b
    np.testing.assert_array_equal(c.measurements['q0'], np.array([[0, 1], [1, 0], [0, 1], [0, 1]]))
    np.testing.assert_array_equal(c.measurements['q1'], np.array([[0], [0], [1], [0]]))
    a = cirq.ResultDict(params=cirq.ParamResolver({'ax': 1}), records={'q0': np.array([[[0, 1]], [[1, 0]], [[0, 1]]], dtype=bool), 'q1': np.array([[[0], [0]], [[0], [1]], [[1], [0]]], dtype=bool)})
    b = cirq.ResultDict(params=cirq.ParamResolver({'ax': 1}), records={'q0': np.array([[[0, 1]]], dtype=bool), 'q1': np.array([[[1], [1]]], dtype=bool)})
    c = a + b
    np.testing.assert_array_equal(c.records['q0'], np.array([[[0, 1]], [[1, 0]], [[0, 1]], [[0, 1]]]))
    np.testing.assert_array_equal(c.records['q1'], np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]]))