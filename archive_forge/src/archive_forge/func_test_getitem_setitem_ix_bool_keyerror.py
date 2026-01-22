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
@pytest.mark.parametrize('bool_value', [True, False])
def test_getitem_setitem_ix_bool_keyerror(self, bool_value):
    df = DataFrame({'a': [1, 2, 3]})
    message = f'{bool_value}: boolean label can not be used without a boolean index'
    with pytest.raises(KeyError, match=message):
        df.loc[bool_value]
    msg = 'cannot use a single bool to index into setitem'
    with pytest.raises(KeyError, match=msg):
        df.loc[bool_value] = 0