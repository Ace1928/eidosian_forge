import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
@pytest.mark.skipif(Engine.get() not in ('Ray', 'Unidist', 'Dask'), reason='We only need to test this case where sort does not default to pandas.')
def test_sort_values_with_sort_key_on_partition_boundary():
    modin_df = pd.DataFrame(np.random.rand(1000, 100), columns=[f'col {i}' for i in range(100)])
    sort_key = modin_df.columns[modin_df._query_compiler._modin_frame.column_widths[0]]
    eval_general(modin_df, modin_df._to_pandas(), lambda df: df.sort_values(sort_key))