import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_swaplevel():
    data = np.random.randint(1, 100, 12)
    modin_df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples([(num, letter, color) for num in range(1, 3) for letter in ['a', 'b', 'c'] for color in ['Red', 'Green']], names=['Number', 'Letter', 'Color']))
    pandas_df = pandas.DataFrame(data, index=pandas.MultiIndex.from_tuples([(num, letter, color) for num in range(1, 3) for letter in ['a', 'b', 'c'] for color in ['Red', 'Green']], names=['Number', 'Letter', 'Color']))
    df_equals(modin_df.swaplevel('Number', 'Color'), pandas_df.swaplevel('Number', 'Color'))
    df_equals(modin_df.swaplevel(), pandas_df.swaplevel())
    df_equals(modin_df.swaplevel(0, 1), pandas_df.swaplevel(0, 1))