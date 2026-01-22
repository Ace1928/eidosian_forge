from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_align_asfreq_method_raises(self):
    df = DataFrame({'A': [1, np.nan, 2]})
    msg = 'Invalid fill method'
    msg2 = "The 'method', 'limit', and 'fill_axis' keywords"
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            df.align(df.iloc[::-1], method='asfreq')