import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('dropna, idx, outputs', [(True, ['A', 'B'], {'b': [123.23, 13.0], 'c': [123.0, 13.0], 'd': [1.0, 13.0]}), (False, ['A', 'B', np.nan], {'b': [123.23, 13.0, 12.3], 'c': [123.0, 13.0, 233.0], 'd': [1.0, 13.0, 12.0]})])
def test_groupby_dropna_normal_index_dataframe(dropna, idx, outputs):
    df_list = [['B', 12, 12, 12], [None, 12.3, 233.0, 12], ['A', 123.23, 123, 1], ['B', 1, 1, 1.0]]
    df = pd.DataFrame(df_list, columns=['a', 'b', 'c', 'd'])
    grouped = df.groupby('a', dropna=dropna).sum()
    expected = pd.DataFrame(outputs, index=pd.Index(idx, dtype='object', name='a'))
    tm.assert_frame_equal(grouped, expected)