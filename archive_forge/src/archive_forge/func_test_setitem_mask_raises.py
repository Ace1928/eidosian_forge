import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
def test_setitem_mask_raises(self, data, box_in_series, request):
    if not box_in_series:
        mark = pytest.mark.xfail(reason='Fails to raise')
        request.applymarker(mark)
    super().test_setitem_mask_raises(data, box_in_series)