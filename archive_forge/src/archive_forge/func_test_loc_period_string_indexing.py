import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_period_string_indexing():
    a = pd.period_range('2013Q1', '2013Q4', freq='Q')
    i = (1111, 2222, 3333)
    idx = MultiIndex.from_product((a, i), names=('Period', 'CVR'))
    df = DataFrame(index=idx, columns=('OMS', 'OMK', 'RES', 'DRIFT_IND', 'OEVRIG_IND', 'FIN_IND', 'VARE_UD', 'LOEN_UD', 'FIN_UD'))
    result = df.loc[('2013Q1', 1111), 'OMS']
    alt = df.loc[(a[0], 1111), 'OMS']
    assert np.isnan(alt)
    assert np.isnan(result)
    alt = df.loc[('2013Q1', 1111), 'OMS']
    assert np.isnan(alt)