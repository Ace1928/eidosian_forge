from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('method', ['strip', 'rstrip', 'lstrip'])
def test_str_attribute(self, method):
    index = Index([' jack', 'jill ', ' jesse ', 'frank'])
    expected = Index([getattr(str, method)(x) for x in index.values])
    result = getattr(index.str, method)()
    tm.assert_index_equal(result, expected)