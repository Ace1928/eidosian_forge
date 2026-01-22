import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_missing_attribute(self, df):
    message = "module 'pandas' has no attribute 'thing'"
    with pytest.raises(AttributeError, match=message):
        df.eval('@pd.thing')