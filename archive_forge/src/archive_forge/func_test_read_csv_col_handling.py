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
@pytest.mark.parametrize('engine', [None, 'arrow'])
@pytest.mark.parametrize('usecols', [None, ['col1'], ['col1', 'col1'], ['col1', 'col2', 'col6'], ['col6', 'col2', 'col1'], [0], [0, 0], [0, 1, 5], [5, 1, 0], lambda x: x in ['col1', 'col2']])
def test_read_csv_col_handling(self, engine, usecols):
    eval_io(fn_name='read_csv', check_kwargs_callable=not callable(usecols), md_extra_kwargs={'engine': engine}, filepath_or_buffer=pytest.csvs_names['test_read_csv_regular'], usecols=usecols)