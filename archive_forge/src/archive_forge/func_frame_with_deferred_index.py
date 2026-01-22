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
def frame_with_deferred_index(self):
    df = pd.DataFrame(**self._df_kwargs)
    try:
        df._query_compiler._modin_frame.set_index_cache(None)
    except AttributeError:
        pytest.skip(reason="Selected execution doesn't support deferred indices.")
    return df