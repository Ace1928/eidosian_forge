import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_getitem_scalar(self, data):
    result = data[0]
    assert isinstance(result, data.dtype.type)
    result = pd.Series(data)[0]
    assert isinstance(result, data.dtype.type)