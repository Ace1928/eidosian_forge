import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('string', [0, 3.14, ('a', 'b'), None])
def test_construction_from_string_errors(self, string):
    msg = f"'construct_from_string' expects a string, got {type(string)}"
    with pytest.raises(TypeError, match=re.escape(msg)):
        IntervalDtype.construct_from_string(string)