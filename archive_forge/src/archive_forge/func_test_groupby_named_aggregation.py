import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
def test_groupby_named_aggregation():
    modin_ser, pandas_ser = create_test_series([10, 10, 10, 1, 1, 1, 2, 3], name='data')
    eval_general(modin_ser, pandas_ser, lambda ser: ser.groupby(level=0).agg(result='max'))