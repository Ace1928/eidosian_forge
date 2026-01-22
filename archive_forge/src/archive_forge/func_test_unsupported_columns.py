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
@pytest.mark.parametrize('data,is_good', [[['1', '2', None, '2', '1'], True], [[None, '3', None, '2', '1'], True], [[1, '2', None, '2', '1'], False], [[None, 3, None, '2', '1'], False]])
def test_unsupported_columns(self, data, is_good):
    pandas_df = pandas.DataFrame({'col': data})
    bad_cols = HdkOnNativeDataframePartitionManager._get_unsupported_cols(pandas_df)
    if is_good:
        assert not bad_cols
    else:
        assert bad_cols == ['col']