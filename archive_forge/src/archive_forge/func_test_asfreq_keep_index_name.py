from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq_keep_index_name(self, frame_or_series):
    index_name = 'bar'
    index = date_range('20130101', periods=20, name=index_name)
    obj = DataFrame(list(range(20)), columns=['foo'], index=index)
    obj = tm.get_obj(obj, frame_or_series)
    assert index_name == obj.index.name
    assert index_name == obj.asfreq('10D').index.name