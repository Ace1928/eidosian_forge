import numpy as np
from numpy.testing import assert_equal, assert_raises
from pandas import Series
import pytest
from statsmodels.graphics.factorplots import _recode, interaction_plot
def test_recode_series(self):
    series = Series(['a', 'b'] * 10, index=np.arange(0, 40, 2), name='index_test')
    series_ = _recode(series, {'a': 0, 'b': 1})
    assert_equal(series_.index.values, series.index.values, err_msg='_recode changed the index')