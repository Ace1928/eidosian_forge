import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('method', ['median', 'skew', 'std', 'var', 'sem'])
def test_median_skew_std_var_sem_1953(method):
    arrays = [['1', '1', '2', '2'], ['1', '2', '3', '4']]
    data = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    modin_df = pd.DataFrame(data, index=arrays)
    pandas_df = pandas.DataFrame(data, index=arrays)
    eval_general(modin_df, pandas_df, lambda df: getattr(df, method)())