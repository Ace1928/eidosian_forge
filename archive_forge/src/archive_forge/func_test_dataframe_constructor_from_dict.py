import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock
@pytest.mark.parametrize('from_series', [True, False])
def test_dataframe_constructor_from_dict(self, data, from_series):
    if from_series:
        data = pd.Series(data)
    result = pd.DataFrame({'A': data})
    assert result.dtypes['A'] == data.dtype
    assert result.shape == (len(data), 1)
    if hasattr(result._mgr, 'blocks'):
        assert isinstance(result._mgr.blocks[0], EABackedBlock)
    assert isinstance(result._mgr.arrays[0], ExtensionArray)