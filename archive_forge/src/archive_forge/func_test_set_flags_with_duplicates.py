import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('cls, axes', [(pd.Series, {'index': ['a', 'a'], 'dtype': float}), (pd.DataFrame, {'index': ['a', 'a']}), (pd.DataFrame, {'index': ['a', 'a'], 'columns': ['b', 'b']}), (pd.DataFrame, {'columns': ['b', 'b']})])
def test_set_flags_with_duplicates(self, cls, axes):
    result = cls(**axes)
    assert result.flags.allows_duplicate_labels is True
    msg = 'Index has duplicates.'
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        cls(**axes).set_flags(allows_duplicate_labels=False)