import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_freq_validation(self):
    arr = np.arange(5, dtype=np.int64) * 3600 * 10 ** 9
    msg = 'Inferred frequency h from passed values does not conform to passed frequency W-SUN'
    depr_msg = 'DatetimeArray.__init__ is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        with pytest.raises(ValueError, match=msg):
            DatetimeArray(arr, freq='W')