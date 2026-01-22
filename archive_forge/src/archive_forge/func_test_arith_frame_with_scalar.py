import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
    if len(data[0]) != 1:
        mark = pytest.mark.xfail(reason='raises in coercing to Series')
        request.applymarker(mark)
    super().test_arith_frame_with_scalar(data, all_arithmetic_operators)