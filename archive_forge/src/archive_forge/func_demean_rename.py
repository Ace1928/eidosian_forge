import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def demean_rename(x):
    result = x - x.mean()
    if isinstance(x, Series):
        return result
    result = result.rename(columns={c: f'{c}_demeaned' for c in result.columns})
    return result