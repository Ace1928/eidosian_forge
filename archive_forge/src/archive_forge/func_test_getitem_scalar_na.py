import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_getitem_scalar_na(self, data_missing, na_cmp, na_value):
    result = data_missing[0]
    assert na_cmp(result, na_value)