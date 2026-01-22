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
def test_groupby_getitem_preserves_key_order_issue_6154():
    a = np.tile(['a', 'b', 'c', 'd', 'e'], (1, 10))
    np.random.shuffle(a[0])
    df = pd.DataFrame(np.hstack((a.T, np.arange(100).reshape((50, 2)))), columns=['col 1', 'col 2', 'col 3'])
    eval_general(df, df._to_pandas(), lambda df: df.groupby('col 1')[['col 3', 'col 2']].count())