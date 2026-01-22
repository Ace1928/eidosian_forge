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
def test_getitem_segfault_with_empty_like_object(self):
    df = DataFrame(np.empty((1, 1), dtype=object))
    df[0] = np.empty_like(df[0])
    df[[0]]