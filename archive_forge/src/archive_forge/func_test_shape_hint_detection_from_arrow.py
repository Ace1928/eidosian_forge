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
def test_shape_hint_detection_from_arrow(self):
    at = pyarrow.Table.from_pydict({'a': [1, 2, 3]})
    df = pd.utils.from_arrow(at)
    assert df._query_compiler._shape_hint == 'column'
    at = pyarrow.Table.from_pydict({'a': [1], 'b': [2], 'c': [3]})
    df = pd.utils.from_arrow(at)
    assert df._query_compiler._shape_hint == 'row'
    at = pyarrow.Table.from_pydict({'a': [1, 2, 3], 'b': [1, 2, 3]})
    df = pd.utils.from_arrow(at)
    assert df._query_compiler._shape_hint is None
    at = pyarrow.Table.from_pydict({'a': [1]})
    df = pd.utils.from_arrow(at)
    assert df._query_compiler._shape_hint == 'column'