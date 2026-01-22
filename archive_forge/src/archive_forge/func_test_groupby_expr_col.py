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
def test_groupby_expr_col(self):

    def groupby(df, **kwargs):
        df = df.loc[:, ['b', 'c']]
        df['year'] = df['c'].dt.year
        df['month'] = df['c'].dt.month
        df['id1'] = df['year'] * 12 + df['month']
        df['id2'] = (df['id1'] - 24000) // 12
        df = df.groupby(['id1', 'id2'], as_index=False).agg({'b': 'max'})
        return df
    run_and_compare(groupby, data=self.taxi_data)