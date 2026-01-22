import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('dropna, expected', [(True, pd.Series([210.0, 350.0], index=['a', 'b'], name='Max Speed')), (False, pd.Series([210.0, 350.0, 20.0], index=['a', 'b', np.nan], name='Max Speed'))])
def test_groupby_dropna_series_by(dropna, expected):
    ser = pd.Series([390.0, 350.0, 30.0, 20.0], index=['Falcon', 'Falcon', 'Parrot', 'Parrot'], name='Max Speed')
    result = ser.groupby(['a', 'b', 'a', np.nan], dropna=dropna).mean()
    tm.assert_series_equal(result, expected)