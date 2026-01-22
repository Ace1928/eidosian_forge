import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('data', [pd.Series(index=[0, 0], dtype=float), pd.DataFrame(index=[0, 0]), pd.DataFrame(columns=[0, 0])])
def test_setting_allows_duplicate_labels_raises(self, data):
    msg = 'Index has duplicates.'
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        data.flags.allows_duplicate_labels = False
    assert data.flags.allows_duplicate_labels is True