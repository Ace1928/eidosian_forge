import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('func', [list, pd.Index, pd.Series, np.array])
def test_groupby_duplicated_columns(func):
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 3], 'C': ['G', 'G']})
    result = df.groupby('C')[func(['A', 'B', 'A'])].mean()
    expected = pd.DataFrame([[1.5, 3.0, 1.5]], columns=['A', 'B', 'A'], index=pd.Index(['G'], name='C'))
    tm.assert_frame_equal(result, expected)