import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('kind', ['bar', 'line'])
def test_dictionary_color(self, kind):
    data_files = ['a', 'b']
    expected = [(0.5, 0.24, 0.6), (0.3, 0.7, 0.7)]
    df1 = DataFrame(np.random.default_rng(2).random((2, 2)), columns=data_files)
    dic_color = {'b': (0.3, 0.7, 0.7), 'a': (0.5, 0.24, 0.6)}
    ax = df1.plot(kind=kind, color=dic_color)
    if kind == 'bar':
        colors = [rect.get_facecolor()[0:-1] for rect in ax.get_children()[0:3:2]]
    else:
        colors = [rect.get_color() for rect in ax.get_lines()[0:2]]
    assert all((color == expected[index] for index, color in enumerate(colors)))