from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.missing import NA
from pandas.core.dtypes.common import is_scalar
import pandas as pd
import pandas._testing as tm
def test_pickle_roundtrip_pandas():
    result = tm.round_trip_pickle(NA)
    assert result is NA