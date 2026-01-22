import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('start', [None, 0, 1, 10, -1, -10])
@pytest.mark.parametrize('stop', [None, 0, 1, 10, -1, -10])
@pytest.mark.parametrize('step', [None, 1, 5])
def test_against_df_iloc(start, stop, step):
    n_rows = 30
    data = {'group': ['group 0'] * n_rows, 'value': list(range(n_rows))}
    df = pd.DataFrame(data)
    grouped = df.groupby('group', as_index=False)
    result = grouped._positional_selector[start:stop:step]
    expected = df.iloc[start:stop:step]
    tm.assert_frame_equal(result, expected)