from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_corner2(self):
    data = {'title': ['foobar', 'bar', 'foobar'] + ['foobar'] * 17, 'cruft': np.random.default_rng(2).random(20)}
    df = DataFrame(data)
    ix = df[df['title'] == 'bar'].index
    df.loc[ix, ['title']] = 'foobar'
    df.loc[ix, ['cruft']] = 0
    assert df.loc[1, 'title'] == 'foobar'
    assert df.loc[1, 'cruft'] == 0