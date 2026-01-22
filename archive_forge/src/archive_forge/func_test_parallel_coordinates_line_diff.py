import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
def test_parallel_coordinates_line_diff(self, iris):
    from pandas.plotting import parallel_coordinates
    df = iris
    ax = _check_plot_works(parallel_coordinates, frame=df, class_column='Name')
    nlines = len(ax.get_lines())
    nxticks = len(ax.xaxis.get_ticklabels())
    ax = _check_plot_works(parallel_coordinates, frame=df, class_column='Name', axvlines=False)
    assert len(ax.get_lines()) == nlines - nxticks