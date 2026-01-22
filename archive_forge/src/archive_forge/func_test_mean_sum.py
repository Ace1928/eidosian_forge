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
def test_mean_sum(self):
    all_codes = np.typecodes['All']
    exclude_codes = np.typecodes['Datetime'] + np.typecodes['Complex'] + 'gSUVO'
    supported_codes = set(all_codes) - set(exclude_codes)

    def test(df, dtype_code, operation, **kwargs):
        df = type(df)({'A': [0, 1], 'B': [1, 0]}, dtype=np.dtype(dtype_code))
        return getattr(df, operation)()
    for c in supported_codes:
        for op in ('sum', 'mean'):
            run_and_compare(test, data={}, dtype_code=c, operation=op)