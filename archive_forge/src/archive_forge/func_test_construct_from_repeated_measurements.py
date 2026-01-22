import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_construct_from_repeated_measurements():
    r = cirq.ResultDict(params=None, records={'a': np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]]), 'b': np.array([[[0, 0, 0]], [[1, 1, 1]]])})
    with pytest.raises(ValueError):
        _ = r.measurements
    assert np.all(r.records['a'] == np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]]))
    assert np.all(r.records['b'] == np.array([[[0, 0, 0]], [[1, 1, 1]]]))
    assert r.repetitions == 2
    r2 = cirq.ResultDict(params=None, records={'a': np.array([[[0, 0]], [[1, 1]]]), 'b': np.array([[[0, 0, 0]], [[1, 1, 1]]])})
    assert np.all(r2.measurements['a'] == np.array([[0, 0], [1, 1]]))
    assert np.all(r2.measurements['b'] == np.array([[0, 0, 0], [1, 1, 1]]))
    assert np.all(r2.records['a'] == np.array([[[0, 0]], [[1, 1]]]))
    assert np.all(r2.records['b'] == np.array([[[0, 0, 0]], [[1, 1, 1]]]))
    assert r2.repetitions == 2