import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def xbox2(x):
    if isinstance(x, NumpyExtensionArray):
        return x._ndarray
    if isinstance(x, BooleanArray):
        return x.astype(bool)
    return x