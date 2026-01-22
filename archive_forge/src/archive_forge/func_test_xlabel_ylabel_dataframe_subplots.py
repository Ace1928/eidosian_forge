import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('index_name, old_label, new_label', [(None, '', 'new'), ('old', 'old', 'new'), (None, '', ''), (None, '', 1), (None, '', [1, 2])])
@pytest.mark.parametrize('kind', ['line', 'area', 'bar'])
def test_xlabel_ylabel_dataframe_subplots(self, kind, index_name, old_label, new_label):
    df = DataFrame([[1, 2], [2, 5]], columns=['Type A', 'Type B'])
    df.index.name = index_name
    axes = df.plot(kind=kind, subplots=True)
    assert all((ax.get_ylabel() == '' for ax in axes))
    assert all((ax.get_xlabel() == old_label for ax in axes))
    axes = df.plot(kind=kind, ylabel=new_label, xlabel=new_label, subplots=True)
    assert all((ax.get_ylabel() == str(new_label) for ax in axes))
    assert all((ax.get_xlabel() == str(new_label) for ax in axes))