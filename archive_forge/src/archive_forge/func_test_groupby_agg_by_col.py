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
@pytest.mark.parametrize('by', [['a'], ['a', 'b', 'c']])
@pytest.mark.parametrize('agg', ['sum', 'size', 'mean', 'median'])
@pytest.mark.parametrize('as_index', [True, False])
def test_groupby_agg_by_col(self, by, agg, as_index):

    def simple_agg(df, **kwargs):
        return df.groupby(by, as_index=as_index).agg(agg)
    run_and_compare(simple_agg, data=self.data)

    def dict_agg(df, **kwargs):
        return df.groupby(by, as_index=as_index).agg({by[0]: agg})
    run_and_compare(dict_agg, data=self.data)

    def dict_agg_all_cols(df, **kwargs):
        return df.groupby(by, as_index=as_index).agg({col: agg for col in by})
    run_and_compare(dict_agg_all_cols, data=self.data)