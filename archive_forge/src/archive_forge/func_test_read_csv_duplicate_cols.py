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
@pytest.mark.parametrize('cols', ['c1,c2,c3', 'c1,c1,c2', 'c1,c1,c1.1,c1.2,c1', 'c1,c1,c1,c1.1,c1.2,c1.3', 'c1.1,c1.2,c1.3,c1,c1,c1', 'c1.1,c1,c1.2,c1,c1.3,c1', 'c1,c1.1,c1,c1.2,c1,c1.3', 'c1,c1,c1.1,c1.1,c1.2,c2', 'c1,c1,c1.1,c1.1,c1.2,c1.2,c2', 'c1.1,c1.1,c1,c1,c1.2,c1.2,c2', 'c1.1,c1,c1.1,c1,c1.1,c1.2,c1.2,c2'])
def test_read_csv_duplicate_cols(self, cols):

    def test(df, lib, **kwargs):
        data = f'{cols}\n'
        with ensure_clean('.csv') as fname:
            with open(fname, 'w') as f:
                f.write(data)
            return lib.read_csv(fname)
    run_and_compare(test, data={})