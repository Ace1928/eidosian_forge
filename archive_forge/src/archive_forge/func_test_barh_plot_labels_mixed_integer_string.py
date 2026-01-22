import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_barh_plot_labels_mixed_integer_string(self):
    from matplotlib.text import Text
    df = DataFrame([{'word': 1, 'value': 0}, {'word': 'knowledge', 'value': 2}])
    plot_barh = df.plot.barh(x='word', legend=None)
    expected_yticklabels = [Text(0, 0, '1'), Text(0, 1, 'knowledge')]
    assert all((actual.get_text() == expected.get_text() for actual, expected in zip(plot_barh.get_yticklabels(), expected_yticklabels)))