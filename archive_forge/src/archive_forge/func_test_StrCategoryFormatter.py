import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('ydata', cases, ids=ids)
def test_StrCategoryFormatter(self, ydata):
    unit = cat.UnitData(ydata)
    labels = cat.StrCategoryFormatter(unit._mapping)
    for i, d in enumerate(ydata):
        assert labels(i, i) == d
        assert labels(i, None) == d