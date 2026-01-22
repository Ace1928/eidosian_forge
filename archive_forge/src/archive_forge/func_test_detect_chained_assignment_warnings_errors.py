from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_detect_chained_assignment_warnings_errors(self, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'A': ['aaa', 'bbb', 'ccc'], 'B': [1, 2, 3]})
    if using_copy_on_write or warn_copy_on_write:
        with tm.raises_chained_assignment_error():
            df.loc[0]['A'] = 111
        return
    with option_context('chained_assignment', 'warn'):
        with tm.assert_produces_warning(SettingWithCopyWarning):
            df.loc[0]['A'] = 111
    with option_context('chained_assignment', 'raise'):
        with pytest.raises(SettingWithCopyError, match=msg):
            df.loc[0]['A'] = 111