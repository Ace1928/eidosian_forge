from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.arm_slow
def test_detect_chained_assignment_doc_example(self, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': ['one', 'one', 'two', 'three', 'two', 'one', 'six'], 'c': Series(range(7), dtype='int64')})
    assert df._is_copy is None
    indexer = df.a.str.startswith('o')
    if using_copy_on_write or warn_copy_on_write:
        with tm.raises_chained_assignment_error():
            df[indexer]['c'] = 42
    else:
        with pytest.raises(SettingWithCopyError, match=msg):
            df[indexer]['c'] = 42