from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_nunique_ints(index_or_series_or_array):
    values = index_or_series_or_array(np.random.default_rng(2).integers(0, 20, 30))
    result = algos.nunique_ints(values)
    expected = len(algos.unique(values))
    assert result == expected