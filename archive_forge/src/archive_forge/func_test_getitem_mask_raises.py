import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_getitem_mask_raises(self, data):
    mask = np.array([True, False])
    msg = f'Boolean index has wrong length: 2 instead of {len(data)}'
    with pytest.raises(IndexError, match=msg):
        data[mask]
    mask = pd.array(mask, dtype='boolean')
    with pytest.raises(IndexError, match=msg):
        data[mask]