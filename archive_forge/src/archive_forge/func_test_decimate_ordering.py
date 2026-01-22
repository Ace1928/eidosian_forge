import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_decimate_ordering(self):
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    curve = Curve((x, y))
    decimated = decimate(curve, max_samples=20)
    renderer('bokeh').get_plot(decimated)
    index = decimated.data[()].data.index
    assert np.all(index == np.sort(index))