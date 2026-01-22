import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
def test_qcut_all_bins_same():
    with pytest.raises(ValueError, match='edges.*unique'):
        qcut([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3)