from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_plot_figsize_and_title(self, series):
    _, ax = mpl.pyplot.subplots()
    ax = series.plot(title='Test', figsize=(16, 8), ax=ax)
    _check_text_labels(ax.title, 'Test')
    _check_axes_shape(ax, axes_num=1, layout=(1, 1), figsize=(16, 8))