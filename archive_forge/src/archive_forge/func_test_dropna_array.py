import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_dropna_array(self, data_missing):
    result = data_missing.dropna()
    expected = data_missing[[1]]
    tm.assert_extension_array_equal(result, expected)