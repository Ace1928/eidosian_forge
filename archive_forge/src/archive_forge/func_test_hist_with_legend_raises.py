import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('by', [None, 'c'])
@pytest.mark.parametrize('column', [None, 'b'])
def test_hist_with_legend_raises(self, by, column):
    index = Index(15 * ['1'] + 15 * ['2'], name='c')
    df = DataFrame(np.random.default_rng(2).standard_normal((30, 2)), index=index, columns=['a', 'b'])
    with pytest.raises(ValueError, match='Cannot use both legend and label'):
        df.hist(legend=True, by=by, column=column, label='d')