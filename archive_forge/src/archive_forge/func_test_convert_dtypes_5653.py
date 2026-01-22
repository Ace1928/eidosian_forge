import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.pandas.testing import assert_index_equal, assert_series_equal
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_convert_dtypes_5653():
    modin_part1 = pd.DataFrame({'col1': ['a', 'b', 'c', 'd']})
    modin_part2 = pd.DataFrame({'col1': [None, None, None, None]})
    modin_df = pd.concat([modin_part1, modin_part2])
    if StorageFormat.get() == 'Pandas':
        assert modin_df._query_compiler._modin_frame._partitions.shape == (2, 1)
    modin_df = modin_df.convert_dtypes()
    assert len(modin_df.dtypes) == 1
    assert modin_df.dtypes.iloc[0] == 'string'