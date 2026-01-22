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
def test_reindex_no_type_preserve_target_empty_mi(self):
    index = Index(list('abc'))
    result = index.reindex(MultiIndex([Index([], np.int64), Index([], np.float64)], [[], []]))[0]
    assert result.levels[0].dtype.type == np.int64
    assert result.levels[1].dtype.type == np.float64