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
@pytest.mark.parametrize('null_dtype', ['category', 'float64'])
def test_null_col(self, null_dtype):
    csv_file = os.path.join(self.root, 'modin/tests/pandas/data', 'test_null_col.csv')
    ref = pandas.read_csv(csv_file, names=['a', 'b', 'c'], dtype={'a': 'int64', 'b': 'int64', 'c': null_dtype}, skiprows=1)
    ref['a'] = ref['a'] + ref['b']
    exp = pd.read_csv(csv_file, names=['a', 'b', 'c'], dtype={'a': 'int64', 'b': 'int64', 'c': null_dtype}, skiprows=1)
    exp['a'] = exp['a'] + exp['b']
    if null_dtype == 'category':
        ref['c'] = ref['c'].astype('string')
        with ForceHdkImport(exp):
            exp = to_pandas(exp)
        exp['c'] = exp['c'].astype('string')
    df_equals(ref, exp)