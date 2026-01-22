from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_msg(self):
    idx = period_range('2000-1-1', freq='Y', periods=10)
    bad_period = Period('2012', 'Y')
    with pytest.raises(KeyError, match="^Period\\('2012', 'Y-DEC'\\)$"):
        idx.get_loc(bad_period)
    try:
        idx.get_loc(bad_period)
    except KeyError as inst:
        assert inst.args[0] == bad_period