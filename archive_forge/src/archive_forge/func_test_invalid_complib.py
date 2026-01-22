import datetime
from io import BytesIO
import re
import numpy as np
import pytest
from pandas import (
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import (
def test_invalid_complib(setup_path):
    df = DataFrame(np.random.default_rng(2).random((4, 5)), index=list('abcd'), columns=list('ABCDE'))
    with tm.ensure_clean(setup_path) as path:
        msg = 'complib only supports \\[.*\\] compression.'
        with pytest.raises(ValueError, match=msg):
            df.to_hdf(path, key='df', complib='foolib')