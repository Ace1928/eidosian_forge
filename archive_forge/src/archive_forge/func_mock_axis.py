import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
@pytest.fixture(autouse=True)
def mock_axis(self, request):
    self.cc = cat.StrCategoryConverter()
    self.unit = cat.UnitData()
    self.ax = FakeAxis(self.unit)