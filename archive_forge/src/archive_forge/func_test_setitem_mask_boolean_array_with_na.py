import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.xfail(reason='cannot set using a list-like indexer with a different length')
def test_setitem_mask_boolean_array_with_na(self, data, box_in_series):
    super().test_setitem_mask_boolean_array_with_na(data, box_in_series)