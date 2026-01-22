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
def test_reductions_skipna_none_raises(self, request, frame_or_series, all_reductions):
    if all_reductions == 'count':
        request.applymarker(pytest.mark.xfail(reason='Count does not accept skipna'))
    obj = frame_or_series([1, 2, 3])
    msg = 'For argument "skipna" expected type bool, received type NoneType.'
    with pytest.raises(ValueError, match=msg):
        getattr(obj, all_reductions)(skipna=None)