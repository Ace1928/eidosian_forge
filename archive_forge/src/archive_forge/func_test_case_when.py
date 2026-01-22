from __future__ import annotations
import datetime
import itertools
import json
import unittest.mock as mock
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
from pandas.core.indexing import IndexingError
from pandas.errors import SpecificationError
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_series_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution, try_cast_to_pandas
from .utils import (
@pytest.mark.parametrize('base', [pandas.Series(range(40)), pandas.Series([0, 7, 8, 9] * 10, name='c', index=range(0, 80, 2))])
@pytest.mark.parametrize('caselist', _case_when_caselists())
@pytest.mark.skipif(Engine.get() == 'Dask', reason='https://github.com/modin-project/modin/issues/7148')
def test_case_when(base, caselist):
    pandas_result = base.case_when(caselist)
    modin_bases = [pd.Series(base)]
    if StorageFormat.get() != 'Hdk' and f'{StorageFormat.get()}On{Engine.get()}' != 'BaseOnPython':
        modin_base_repart = construct_modin_df_by_scheme(base.to_frame(), partitioning_scheme={'row_lengths': [14, 14, 12], 'column_widths': [1]}).squeeze(axis=1)
        assert modin_bases[0]._query_compiler._modin_frame._partitions.shape != modin_base_repart._query_compiler._modin_frame._partitions.shape
        modin_base_repart.name = base.name
        modin_bases.append(modin_base_repart)
    for modin_base in modin_bases:
        df_equals(pandas_result, modin_base.case_when(caselist))
        if any((isinstance(data, pandas.Series) for case_tuple in caselist for data in case_tuple)):
            caselist = [tuple((pd.Series(data) if isinstance(data, pandas.Series) else data for data in case_tuple)) for case_tuple in caselist]
            df_equals(pandas_result, modin_base.case_when(caselist))