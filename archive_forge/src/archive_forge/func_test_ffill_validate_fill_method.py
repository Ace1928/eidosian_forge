import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('invalid_method', ['linear', 'carrot'])
def test_ffill_validate_fill_method(self, left, right, invalid_method):
    with pytest.raises(ValueError, match=re.escape("fill_method must be 'ffill' or None")):
        merge_ordered(left, right, on='key', fill_method=invalid_method)