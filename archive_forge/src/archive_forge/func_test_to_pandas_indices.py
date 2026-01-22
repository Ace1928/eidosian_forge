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
@pytest.mark.parametrize('data', [test_data_values[0], []], ids=['test_data_values[0]', '[]'])
def test_to_pandas_indices(data):
    md_df = pd.DataFrame(data)
    index = pandas.MultiIndex.from_tuples([(i, i * 2) for i in np.arange(len(md_df) + 1)], names=['A', 'B']).drop(0)
    columns = pandas.MultiIndex.from_tuples([(i, i * 2) for i in np.arange(len(md_df.columns) + 1)], names=['A', 'B']).drop(0)
    md_df.index = index
    md_df.columns = columns
    pd_df = md_df._to_pandas()
    for axis in [0, 1]:
        assert md_df.axes[axis].equals(pd_df.axes[axis]), f'Indices at axis {axis} are different!'
        assert not hasattr(md_df.axes[axis], 'equal_levels') or md_df.axes[axis].equal_levels(pd_df.axes[axis]), f'Levels of indices at axis {axis} are different!'