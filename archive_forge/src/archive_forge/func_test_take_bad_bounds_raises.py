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
def test_take_bad_bounds_raises(self):
    index = Index(list('ABC'), name='xxx')
    with pytest.raises(IndexError, match='out of bounds'):
        index.take(np.array([1, -5]))