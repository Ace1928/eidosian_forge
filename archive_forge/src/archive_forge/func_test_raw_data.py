import os
import re
import numpy as np
import pandas
import pyarrow
import pytest
from pandas._testing import ensure_clean
from pandas.core.dtypes.common import is_list_like
from pyhdk import __version__ as hdk_version
from modin.config import StorageFormat
from modin.tests.interchange.dataframe_protocol.hdk.utils import split_df_into_chunks
from modin.tests.pandas.utils import (
from .utils import ForceHdkImport, eval_io, run_and_compare, set_execution_mode
import modin.pandas as pd
from modin.experimental.core.execution.native.implementations.hdk_on_native.calcite_serializer import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.df_algebra import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.partitioning.partition_manager import (
from modin.pandas.io import from_arrow
from modin.tests.pandas.utils import (
from modin.utils import try_cast_to_pandas
@pytest.mark.parametrize('data', [None, {'A': range(10)}, pandas.Series(range(10)), pandas.DataFrame({'A': range(10)})])
@pytest.mark.parametrize('index', [None, pandas.RangeIndex(10), pandas.RangeIndex(start=10, stop=0, step=-1)])
@pytest.mark.parametrize('columns', [None, ['A'], ['A', 'B', 'C']])
@pytest.mark.parametrize('dtype', [None, float])
def test_raw_data(self, data, index, columns, dtype):
    if isinstance(data, pandas.Series) and data.name is None and (columns is not None) and (len(columns) > 1):
        data = data.copy()
        data.name = 'D'
    mdf, pdf = create_test_dfs(data, index=index, columns=columns, dtype=dtype)
    df_equals(mdf, pdf)