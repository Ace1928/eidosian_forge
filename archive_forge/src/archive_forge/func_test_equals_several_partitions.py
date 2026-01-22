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
def test_equals_several_partitions():
    modin_series1 = pd.concat([pd.DataFrame([0, 1]), pd.DataFrame([None, 1])])
    modin_series2 = pd.concat([pd.DataFrame([0, 1]), pd.DataFrame([1, None])])
    assert not modin_series1.equals(modin_series2)