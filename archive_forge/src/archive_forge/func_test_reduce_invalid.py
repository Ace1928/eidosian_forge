from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_reduce_invalid(self, arr1d):
    msg = "does not support reduction 'not a method'"
    with pytest.raises(TypeError, match=msg):
        arr1d._reduce('not a method')