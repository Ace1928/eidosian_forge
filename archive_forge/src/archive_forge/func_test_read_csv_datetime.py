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
@pytest.mark.parametrize('parse_dates', [True, False, ['col2'], ['c2'], [['col2', 'col3']], {'col23': ['col2', 'col3']}, []])
@pytest.mark.parametrize('names', [None, [f'c{x}' for x in range(1, 7)]])
def test_read_csv_datetime(self, engine, parse_dates, names, request):
    parse_dates_unsupported = isinstance(parse_dates, dict) or (isinstance(parse_dates, list) and any((not isinstance(date, str) for date in parse_dates)))
    if parse_dates_unsupported and engine == 'arrow' and (not names):
        pytest.skip('In these cases Modin raises `ArrowEngineException` while pandas ' + "doesn't raise any exceptions that causes tests fails")
    skip_exc_type_check = parse_dates_unsupported and engine == 'arrow'
    if skip_exc_type_check:
        pytest.xfail(reason='https://github.com/modin-project/modin/issues/7012')
    expected_exception = None
    if 'names1-parse_dates2' in request.node.callspec.id:
        expected_exception = ValueError("Missing column provided to 'parse_dates': 'col2'")
    elif 'names1-parse_dates5-None' in request.node.callspec.id or 'names1-parse_dates4-None' in request.node.callspec.id:
        expected_exception = ValueError("Missing column provided to 'parse_dates': 'col2, col3'")
    elif 'None-parse_dates3' in request.node.callspec.id:
        expected_exception = ValueError("Missing column provided to 'parse_dates': 'c2'")
    eval_io(fn_name='read_csv', md_extra_kwargs={'engine': engine}, expected_exception=expected_exception, filepath_or_buffer=pytest.csvs_names['test_read_csv_regular'], parse_dates=parse_dates, names=names)