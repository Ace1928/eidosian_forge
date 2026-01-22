import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_cumprod_timedelta(self):
    ser = pd.Series([pd.Timedelta(days=1), pd.Timedelta(days=3)])
    with pytest.raises(TypeError, match='cumprod not supported for Timedelta'):
        ser.cumprod()