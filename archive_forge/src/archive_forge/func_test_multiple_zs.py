import os
import tempfile
from unittest import SkipTest
from collections import OrderedDict
import numpy as np
from holoviews import Store
from holoviews.element import RGB, Image
from holoviews.element.comparison import ComparisonTestCase
def test_multiple_zs(self):
    plot = self.ds.hvplot(x='lat', y='lon', z=['temp', 'precip'], dynamic=False)
    assert 'temp' in plot.keys()
    assert 'precip' in plot.keys()
    assert plot['temp'].kdims == ['lat', 'lon']
    assert plot['precip'].kdims == ['lat', 'lon']