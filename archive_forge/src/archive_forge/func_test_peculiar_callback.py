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
def test_peculiar_callback():

    def func(val):
        if not isinstance(val, tuple):
            raise BaseException('Urgh...')
        return val
    pandas_df = pandas.DataFrame({'col': [(0, 1)]})
    pandas_series = pandas_df['col'].apply(func)
    modin_df = pd.DataFrame({'col': [(0, 1)]})
    modin_series = modin_df['col'].apply(func)
    df_equals(modin_series, pandas_series)