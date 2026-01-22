import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.xfail(reason='only integer scalar arrays can be converted')
def test_setitem_2d_values(self, data):
    super().test_setitem_2d_values(data)