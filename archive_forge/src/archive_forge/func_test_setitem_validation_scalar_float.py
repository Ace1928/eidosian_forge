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
@pytest.mark.parametrize('invalid', _invalid_scalars + [True])
@pytest.mark.parametrize('indexer', _indexers)
def test_setitem_validation_scalar_float(self, invalid, float_numpy_dtype, indexer):
    df = DataFrame({'a': [1, 2, None]}, dtype=float_numpy_dtype)
    self._check_setitem_invalid(df, invalid, indexer, FutureWarning)