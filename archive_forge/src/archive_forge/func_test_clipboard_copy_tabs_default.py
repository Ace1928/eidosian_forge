from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.clipboard import (
@pytest.mark.parametrize('sep', ['\t', None, 'default'])
@pytest.mark.parametrize('excel', [True, None, 'default'])
def test_clipboard_copy_tabs_default(self, sep, excel, df, clipboard):
    kwargs = build_kwargs(sep, excel)
    df.to_clipboard(**kwargs)
    assert clipboard.text() == df.to_csv(sep='\t')