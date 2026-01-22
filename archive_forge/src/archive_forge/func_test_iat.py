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
@pytest.mark.skip(reason='Defaulting to Pandas')
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_iat(data):
    modin_df = pd.DataFrame(data)
    with pytest.raises(NotImplementedError):
        modin_df.iat()