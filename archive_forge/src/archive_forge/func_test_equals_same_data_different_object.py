import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
def test_equals_same_data_different_object(self, data, using_copy_on_write, request):
    if using_copy_on_write:
        mark = pytest.mark.xfail(reason='Fails with CoW')
        request.applymarker(mark)
    super().test_equals_same_data_different_object(data)