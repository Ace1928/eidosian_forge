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
@pytest.mark.parametrize('opname', ['any', 'all'])
@pytest.mark.parametrize('axis', [0, 1])
def test_any_all_bool_with_na(self, opname, axis, bool_frame_with_na):
    getattr(bool_frame_with_na, opname)(axis=axis, bool_only=False)