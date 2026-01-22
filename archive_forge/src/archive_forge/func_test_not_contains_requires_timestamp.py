import re
import numpy as np
import pytest
from pandas._libs import index as libindex
import pandas as pd
@pytest.mark.parametrize('scalar', [pd.Timedelta(pd.Timestamp('2016-01-01').asm8.view('m8[ns]')), pd.Timestamp('2016-01-01')._value, pd.Timestamp('2016-01-01').to_pydatetime(), pd.Timestamp('2016-01-01').to_datetime64()])
def test_not_contains_requires_timestamp(self, scalar):
    dti1 = pd.date_range('2016-01-01', periods=3)
    dti2 = dti1.insert(1, pd.NaT)
    dti3 = dti1.insert(3, dti1[0])
    dti4 = pd.date_range('2016-01-01', freq='ns', periods=2000000)
    dti5 = dti4.insert(0, dti4[0])
    msg = '|'.join([re.escape(str(scalar)), re.escape(repr(scalar))])
    for dti in [dti1, dti2, dti3, dti4, dti5]:
        with pytest.raises(TypeError, match=msg):
            scalar in dti._engine
        with pytest.raises(KeyError, match=msg):
            dti._engine.get_loc(scalar)