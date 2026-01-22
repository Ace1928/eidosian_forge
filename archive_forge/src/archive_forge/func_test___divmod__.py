import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('other', [lambda df: 2, lambda df: df])
def test___divmod__(other):
    data = test_data['float_nan_data']
    eval_general(*create_test_dfs(data), lambda df: divmod(df, other(df)))