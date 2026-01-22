import re
import numpy as np
import pytest
from pandas._libs import index as libindex
import pandas as pd
@pytest.mark.parametrize('scalar', [pd.Timestamp(pd.Timedelta(days=42).asm8.view('datetime64[ns]')), pd.Timedelta(days=42)._value, pd.Timedelta(days=42).to_pytimedelta(), pd.Timedelta(days=42).to_timedelta64()])
def test_not_contains_requires_timedelta(self, scalar):
    tdi1 = pd.timedelta_range('42 days', freq='9h', periods=1234)
    tdi2 = tdi1.insert(1, pd.NaT)
    tdi3 = tdi1.insert(3, tdi1[0])
    tdi4 = pd.timedelta_range('42 days', freq='ns', periods=2000000)
    tdi5 = tdi4.insert(0, tdi4[0])
    msg = '|'.join([re.escape(str(scalar)), re.escape(repr(scalar))])
    for tdi in [tdi1, tdi2, tdi3, tdi4, tdi5]:
        with pytest.raises(TypeError, match=msg):
            scalar in tdi._engine
        with pytest.raises(KeyError, match=msg):
            tdi._engine.get_loc(scalar)