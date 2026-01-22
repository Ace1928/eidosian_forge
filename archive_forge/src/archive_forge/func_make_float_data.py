import warnings
import numpy as np
import pytest
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import (
from pandas.core.arrays.integer import (
from pandas.tests.extension import base
def make_float_data():
    return list(np.arange(0.1, 0.9, 0.1)) + [pd.NA] + list(np.arange(1, 9.8, 0.1)) + [pd.NA] + [9.9, 10.0]