import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.xfail(reason='broadcasting error')
def test_where_series(self, data, na_value):
    super().test_where_series(data, na_value)