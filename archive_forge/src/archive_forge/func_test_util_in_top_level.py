from __future__ import annotations
import pytest
import pandas as pd
from pandas import api
import pandas._testing as tm
from pandas.api import (
def test_util_in_top_level(self):
    with pytest.raises(AttributeError, match='foo'):
        pd.util.foo