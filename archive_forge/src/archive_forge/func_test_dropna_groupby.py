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
@pytest.mark.skip('Dropna logic for GroupBy is disabled for now')
@pytest.mark.parametrize('by', ['col1', ['col1', 'col2'], ['col1', 'col4']])
@pytest.mark.parametrize('dropna', [True, False])
def test_dropna_groupby(self, by, dropna):

    def applier(df, *args, **kwargs):
        return df.groupby(by=by, dropna=dropna).sum().fillna(0)
    run_and_compare(applier, data=self.data)