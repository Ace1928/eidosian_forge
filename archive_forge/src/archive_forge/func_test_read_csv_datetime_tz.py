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
@pytest.mark.parametrize('parse_dates', [None, True, False])
def test_read_csv_datetime_tz(self, engine, parse_dates):
    with ensure_clean('.csv') as file:
        with open(file, 'w') as f:
            f.write('test\n2023-01-01T00:00:00.000-07:00')
        eval_io(fn_name='read_csv', filepath_or_buffer=file, md_extra_kwargs={'engine': engine}, parse_dates=parse_dates)