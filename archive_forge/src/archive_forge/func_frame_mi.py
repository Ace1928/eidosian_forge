import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.fixture
def frame_mi(frame):
    frame.index = MultiIndex.from_product([range(5), range(2)])
    return frame