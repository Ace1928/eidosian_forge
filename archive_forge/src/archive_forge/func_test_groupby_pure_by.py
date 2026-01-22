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
def test_groupby_pure_by(self):
    data = [1, 1, 2, 2]
    run_and_compare(lambda df: df.groupby(df).sum(), data=data, force_lazy=True)
    md_ser, pd_ser = (pd.Series(data), pandas.Series(data))
    md_ser._query_compiler._modin_frame._execute()
    assert isinstance(md_ser._query_compiler._modin_frame._op, FrameNode), "Triggering execution of the Modin frame supposed to set 'FrameNode' as a frame's op"
    set_execution_mode(md_ser, 'lazy')
    md_res = md_ser.groupby(md_ser).sum()
    set_execution_mode(md_res, None)
    pd_res = pd_ser.groupby(pd_ser).sum()
    df_equals(md_res, pd_res)