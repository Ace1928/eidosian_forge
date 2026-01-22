import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
@pytest.mark.xfail(reason='Pandas does not pass this test')
def test_rename_nocopy():
    source_df = pandas.DataFrame(test_data['int_data'])[['col1', 'index', 'col3', 'col4']]
    modin_df = pd.DataFrame(source_df)
    modin_renamed = modin_df.rename(columns={'col3': 'foo'}, copy=False)
    modin_renamed['foo'] = 1
    assert (modin_df['col3'] == 1).all()