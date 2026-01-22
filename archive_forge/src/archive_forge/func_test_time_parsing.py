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
def test_time_parsing(self):
    csv_file = os.path.join(self.root, time_parsing_csv_path)
    for kwargs in ({'skiprows': 1, 'names': ['timestamp', 'year', 'month', 'date', 'symbol', 'high', 'low', 'open', 'close', 'spread', 'volume'], 'parse_dates': ['timestamp'], 'dtype': {'symbol': 'string'}},):
        rp = pandas.read_csv(csv_file, **kwargs)
        rm = pd.read_csv(csv_file, engine='arrow', **kwargs)
        with ForceHdkImport(rm):
            rm = to_pandas(rm)
            df_equals(rm['timestamp'].dt.year, rp['timestamp'].dt.year)
            df_equals(rm['timestamp'].dt.month, rp['timestamp'].dt.month)
            df_equals(rm['timestamp'].dt.day, rp['timestamp'].dt.day)
            df_equals(rm['timestamp'].dt.hour, rp['timestamp'].dt.hour)