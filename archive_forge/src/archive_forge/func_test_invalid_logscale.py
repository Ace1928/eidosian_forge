from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('input_param', ['logx', 'logy', 'loglog'])
def test_invalid_logscale(self, input_param):
    df = DataFrame({'a': np.arange(100)}, index=np.arange(100))
    msg = f"keyword '{input_param}' should be bool, None, or 'sym', not 'sm'"
    with pytest.raises(ValueError, match=msg):
        df.plot(**{input_param: 'sm'})
    msg = f"PiePlot ignores the '{input_param}' keyword"
    with tm.assert_produces_warning(UserWarning, match=msg):
        df.plot.pie(subplots=True, **{input_param: True})