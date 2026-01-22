import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_subplot_titles(self, iris):
    df = iris.drop('Name', axis=1).head()
    title = list(df.columns)
    plot = df.plot(subplots=True, title=title)
    assert [p.get_title() for p in plot] == title