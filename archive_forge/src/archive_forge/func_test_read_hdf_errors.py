import datetime
from io import BytesIO
import re
import numpy as np
import pytest
from pandas import (
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import (
def test_read_hdf_errors(setup_path, tmp_path):
    df = DataFrame(np.random.default_rng(2).random((4, 5)), index=list('abcd'), columns=list('ABCDE'))
    path = tmp_path / setup_path
    msg = 'File [\\S]* does not exist'
    with pytest.raises(OSError, match=msg):
        read_hdf(path, 'key')
    df.to_hdf(path, key='df')
    store = HDFStore(path, mode='r')
    store.close()
    msg = 'The HDFStore must be open for reading.'
    with pytest.raises(OSError, match=msg):
        read_hdf(store, 'df')