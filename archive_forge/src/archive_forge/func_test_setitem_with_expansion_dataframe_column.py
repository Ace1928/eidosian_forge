import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
def test_setitem_with_expansion_dataframe_column(self, data, full_indexer, request):
    if 'full_slice' in request.node.name:
        mark = pytest.mark.xfail(reason='slice is not iterable')
        request.applymarker(mark)
    super().test_setitem_with_expansion_dataframe_column(data, full_indexer)