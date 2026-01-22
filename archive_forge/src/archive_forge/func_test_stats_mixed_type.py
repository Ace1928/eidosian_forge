from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
def test_stats_mixed_type(self, float_string_frame):
    with pytest.raises(TypeError, match='could not convert'):
        float_string_frame.std(1)
    with pytest.raises(TypeError, match='could not convert'):
        float_string_frame.var(1)
    with pytest.raises(TypeError, match='unsupported operand type'):
        float_string_frame.mean(1)
    with pytest.raises(TypeError, match='could not convert'):
        float_string_frame.skew(1)