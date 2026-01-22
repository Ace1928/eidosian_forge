import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('skipna', [True, False])
def test_infer_dtype(self, data, data_missing, skipna):
    res = infer_dtype(data, skipna=skipna)
    assert isinstance(res, str)
    res = infer_dtype(data_missing, skipna=skipna)
    assert isinstance(res, str)