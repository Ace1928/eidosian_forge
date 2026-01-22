from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture(params=[dti, dti.tz_localize('UTC'), dti.to_period('W'), dti - dti[0], rng, pd.Index([1, 2, 3]), pd.Index([2.0, 3.0, 4.0]), pd.Index([4, 5, 6], dtype='u8'), pd.IntervalIndex.from_breaks(dti4)])
def non_comparable_idx(request):
    return request.param