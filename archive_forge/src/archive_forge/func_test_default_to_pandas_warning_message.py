import contextlib
import numpy as np
import pandas
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_frame_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
from .utils import (
@pytest.mark.skipif(get_current_execution() != 'BaseOnPython', reason='This test make sense only on BaseOnPython execution.')
@pytest.mark.parametrize('func, regex', [(lambda df: df.mean(), 'DataFrame\\.mean'), (lambda df: df + df, 'DataFrame\\.add'), (lambda df: df.index, 'DataFrame\\.get_axis\\(0\\)'), (lambda df: df.drop(columns='col1').squeeze().repeat(2), 'Series\\.repeat'), (lambda df: df.groupby('col1').prod(), 'GroupBy\\.prod'), (lambda df: df.rolling(1).count(), 'Rolling\\.count')])
def test_default_to_pandas_warning_message(func, regex):
    data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
    df = pd.DataFrame(data)
    with pytest.warns(UserWarning, match=regex):
        func(df)