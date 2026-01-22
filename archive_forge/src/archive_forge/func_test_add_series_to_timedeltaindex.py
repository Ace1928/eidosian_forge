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
def test_add_series_to_timedeltaindex():
    deltas = pd.to_timedelta([1], unit='h')
    test_series = create_test_series(np.datetime64('2000-12-12'))
    eval_general(*test_series, lambda s: s + deltas)
    eval_general(*test_series, lambda s: s - deltas)