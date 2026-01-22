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
@pytest.mark.parametrize('index', ['string', 'int64', 'int32', 'float64', 'float32'], indirect=True)
@pytest.mark.parametrize('keys', [['foo', 'bar'], ['1', 'bar']])
def test_drop_by_str_label_raises_missing_keys(self, index, keys):
    with pytest.raises(KeyError, match=''):
        index.drop(keys)