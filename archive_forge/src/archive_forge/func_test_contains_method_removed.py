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
def test_contains_method_removed(self, index):
    if isinstance(index, IntervalIndex):
        index.contains(1)
    else:
        msg = f"'{type(index).__name__}' object has no attribute 'contains'"
        with pytest.raises(AttributeError, match=msg):
            index.contains(1)