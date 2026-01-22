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
@pytest.mark.parametrize('empty,klass', [(PeriodIndex([], freq='D'), PeriodIndex), (PeriodIndex(iter([]), freq='D'), PeriodIndex), (PeriodIndex((_ for _ in []), freq='D'), PeriodIndex), (RangeIndex(step=1), RangeIndex), (MultiIndex(levels=[[1, 2], ['blue', 'red']], codes=[[], []]), MultiIndex)])
def test_constructor_empty_special(self, empty, klass):
    assert isinstance(empty, klass)
    assert not len(empty)