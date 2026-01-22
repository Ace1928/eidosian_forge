import datetime
from io import BytesIO
import re
import numpy as np
import pytest
from pandas import (
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import (
def test_pass_spec_to_storer(setup_path):
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
    with ensure_clean_store(setup_path) as store:
        store.put('df', df)
        msg = 'cannot pass a column specification when reading a Fixed format store. this store must be selected in its entirety'
        with pytest.raises(TypeError, match=msg):
            store.select('df', columns=['A'])
        msg = 'cannot pass a where specification when reading from a Fixed format store. this store must be selected in its entirety'
        with pytest.raises(TypeError, match=msg):
            store.select('df', where=['columns=A'])