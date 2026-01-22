import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import is_extension_array_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
import pandas._testing as tm
def test_is_numeric_honored(self, data):
    result = pd.Series(data)
    if hasattr(result._mgr, 'blocks'):
        assert result._mgr.blocks[0].is_numeric is data.dtype._is_numeric