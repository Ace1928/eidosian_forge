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
def test_concat_index_name(self):
    df1 = pandas.DataFrame(self.data)
    df1 = df1.set_index('a')
    df2 = pandas.DataFrame(self.data3)
    df2 = df2.set_index('f')
    ref = pandas.concat([df1, df2], axis=1, join='inner')
    exp = pd.concat([df1, df2], axis=1, join='inner')
    df_equals(ref, exp)
    df2.index.name = 'a'
    ref = pandas.concat([df1, df2], axis=1, join='inner')
    exp = pd.concat([df1, df2], axis=1, join='inner')
    df_equals(ref, exp)