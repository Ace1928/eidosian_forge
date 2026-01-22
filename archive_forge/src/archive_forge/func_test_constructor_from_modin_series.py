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
def test_constructor_from_modin_series(self):

    def construct_has_common_projection(lib, df, **kwargs):
        return lib.DataFrame({'col1': df.iloc[:, 0], 'col2': df.iloc[:, 1]})

    def construct_no_common_projection(lib, df1, df2, **kwargs):
        return lib.DataFrame({'col1': df1.iloc[:, 0], 'col2': df2.iloc[:, 0], 'col3': df1.iloc[:, 1]})

    def construct_mixed_data(lib, df1, df2, **kwargs):
        return lib.DataFrame({'col1': df1.iloc[:, 0], 'col2': df2.iloc[:, 0], 'col3': df1.iloc[:, 1], 'col4': np.arange(len(df1))})
    run_and_compare(construct_has_common_projection, data={'a': [1, 2, 3, 4], 'b': [3, 4, 5, 6]})
    run_and_compare(construct_no_common_projection, data={'a': [1, 2, 3, 4], 'b': [3, 4, 5, 6]}, data2={'a': [10, 20, 30, 40]}, force_lazy=False)
    run_and_compare(construct_mixed_data, data={'a': [1, 2, 3, 4], 'b': [3, 4, 5, 6]}, data2={'a': [10, 20, 30, 40]}, force_lazy=False)