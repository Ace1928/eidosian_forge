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
@pytest.mark.parametrize('klass', [partial(CategoricalIndex, data=[1]), partial(DatetimeIndex, data=['2020-01-01']), partial(PeriodIndex, data=['2020-01-01']), partial(TimedeltaIndex, data=['1 day']), partial(RangeIndex, data=range(1)), partial(IntervalIndex, data=[pd.Interval(0, 1)]), partial(Index, data=['a'], dtype=object), partial(MultiIndex, levels=[1], codes=[0])])
def test_index_subclass_constructor_wrong_kwargs(klass):
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        klass(foo='bar')