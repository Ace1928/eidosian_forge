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
def test_map_with_tuples_mi(self):
    first_level = ['foo', 'bar', 'baz']
    multi_index = MultiIndex.from_tuples(zip(first_level, [1, 2, 3]))
    reduced_index = multi_index.map(lambda x: x[0])
    tm.assert_index_equal(reduced_index, Index(first_level))