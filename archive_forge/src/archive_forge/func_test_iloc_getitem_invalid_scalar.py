from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_iloc_getitem_invalid_scalar(self, frame_or_series):
    obj = DataFrame(np.arange(100).reshape(10, 10))
    obj = tm.get_obj(obj, frame_or_series)
    with pytest.raises(TypeError, match='Cannot index by location index'):
        obj.iloc['a']