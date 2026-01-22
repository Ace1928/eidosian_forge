import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('method, frame_only', [(operator.methodcaller('set_index', 'A', inplace=True), True), (operator.methodcaller('reset_index', inplace=True), True), (operator.methodcaller('rename', lambda x: x, inplace=True), False)])
def test_inplace_raises(method, frame_only):
    df = pd.DataFrame({'A': [0, 0], 'B': [1, 2]}).set_flags(allows_duplicate_labels=False)
    s = df['A']
    s.flags.allows_duplicate_labels = False
    msg = 'Cannot specify'
    with pytest.raises(ValueError, match=msg):
        method(df)
    if not frame_only:
        with pytest.raises(ValueError, match=msg):
            method(s)