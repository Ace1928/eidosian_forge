import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
def test_StrCategoryLocator(self):
    locs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    unit = cat.UnitData([str(j) for j in locs])
    ticks = cat.StrCategoryLocator(unit._mapping)
    np.testing.assert_array_equal(ticks.tick_values(None, None), locs)