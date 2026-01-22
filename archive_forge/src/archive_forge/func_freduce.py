from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def freduce(group):
    assert group.name is not None
    if using_infer_string and grouper == 'A' and is_string_dtype(group.dtype):
        with pytest.raises(TypeError, match='does not support'):
            group.sum()
    else:
        return group.sum()