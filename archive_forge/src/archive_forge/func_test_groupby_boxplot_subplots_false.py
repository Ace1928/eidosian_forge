import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('col, expected_xticklabel', [('v', ['(a, v)', '(b, v)', '(c, v)', '(d, v)', '(e, v)']), (['v'], ['(a, v)', '(b, v)', '(c, v)', '(d, v)', '(e, v)']), ('v1', ['(a, v1)', '(b, v1)', '(c, v1)', '(d, v1)', '(e, v1)']), (['v', 'v1'], ['(a, v)', '(a, v1)', '(b, v)', '(b, v1)', '(c, v)', '(c, v1)', '(d, v)', '(d, v1)', '(e, v)', '(e, v1)']), (None, ['(a, v)', '(a, v1)', '(b, v)', '(b, v1)', '(c, v)', '(c, v1)', '(d, v)', '(d, v1)', '(e, v)', '(e, v1)'])])
def test_groupby_boxplot_subplots_false(self, col, expected_xticklabel):
    df = DataFrame({'cat': np.random.default_rng(2).choice(list('abcde'), 100), 'v': np.random.default_rng(2).random(100), 'v1': np.random.default_rng(2).random(100)})
    grouped = df.groupby('cat')
    axes = _check_plot_works(grouped.boxplot, subplots=False, column=col, return_type='axes')
    result_xticklabel = [x.get_text() for x in axes.get_xticklabels()]
    assert expected_xticklabel == result_xticklabel