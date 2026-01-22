import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
def test_multitype_list_index_access(self):
    df = DataFrame(np.random.default_rng(2).random((10, 5)), columns=['a'] + [20, 21, 22, 23])
    with pytest.raises(KeyError, match=re.escape("'[26, -8] not in index'")):
        df[[22, 26, -8]]
    assert df[21].shape[0] == df.shape[0]