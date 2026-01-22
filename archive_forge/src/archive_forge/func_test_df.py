import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_df():
    result = cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'ab': np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]], dtype=bool), 'c': np.array([[0], [0], [1], [0], [1]], dtype=bool)})
    remove_end_measurements = pd.DataFrame(data={'ab': [1, 1, 2], 'c': [0, 1, 0]}, index=[1, 2, 3], dtype=np.int64)
    pd.testing.assert_frame_equal(result.data.iloc[1:-1], remove_end_measurements)
    df = result.data
    assert len(df[df['ab'] == 1]) == 4
    assert df.c.value_counts().to_dict() == {0: 3, 1: 2}