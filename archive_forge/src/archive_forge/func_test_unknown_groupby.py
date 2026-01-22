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
@pytest.mark.parametrize('columns', [[(False, 'a')], [(False, 'a'), (False, 'b'), (False, 'c')], [(False, 'a'), (False, 'b')], [(False, 'b'), (False, 'a')], [(True, 'a'), (True, 'b'), (True, 'c')], [(True, 'a'), (True, 'b')], [(False, 'a'), (False, 'b'), (True, 'c')], [(False, 'a'), (True, 'c')], [(False, 'a'), (False, pd.Series([5, 6, 7, 8]))]])
def test_unknown_groupby(columns):
    data = {'b': [11, 11, 22, 200], 'c': [111, 111, 222, 7000]}
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    with pytest.raises(KeyError):
        pandas_df.groupby(by=get_external_groupers(pandas_df, columns)[1])
    with pytest.raises(KeyError):
        modin_df.groupby(by=get_external_groupers(modin_df, columns)[1])