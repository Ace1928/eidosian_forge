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
def test_map_defaultdict(self):
    index = Index([1, 2, 3])
    default_dict = defaultdict(lambda: 'blank')
    default_dict[1] = 'stuff'
    result = index.map(default_dict)
    expected = Index(['stuff', 'blank', 'blank'])
    tm.assert_index_equal(result, expected)