import os
import numpy as np
import pytest
from pandas.compat import (
from pandas.errors import (
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io import pytables
from pandas.io.pytables import Term
def test_open_args(setup_path):
    with tm.ensure_clean(setup_path) as path:
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
        store = HDFStore(path, mode='a', driver='H5FD_CORE', driver_core_backing_store=0)
        store['df'] = df
        store.append('df2', df)
        tm.assert_frame_equal(store['df'], df)
        tm.assert_frame_equal(store['df2'], df)
        store.close()
    assert not os.path.exists(path)