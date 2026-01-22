from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_between_time_incorrect_arg_inclusive(self):
    rng = date_range('1/1/2000', '1/5/2000', freq='5min')
    ts = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng)
    stime = time(0, 0)
    etime = time(1, 0)
    inclusive = 'bad_string'
    msg = "Inclusive has to be either 'both', 'neither', 'left' or 'right'"
    with pytest.raises(ValueError, match=msg):
        ts.between_time(stime, etime, inclusive=inclusive)