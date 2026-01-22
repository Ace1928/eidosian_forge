import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('axis', ['rows', 'columns'])
@pytest.mark.parametrize('na_option', ['keep', 'top', 'bottom'], ids=['keep', 'top', 'bottom'])
def test_rank_transposed(axis, na_option):
    eval_general(*create_test_dfs(test_data['int_data']), lambda df: df.rank(axis=axis, na_option=na_option))