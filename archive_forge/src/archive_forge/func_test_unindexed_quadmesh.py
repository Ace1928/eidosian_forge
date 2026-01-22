import os
import tempfile
from unittest import SkipTest
from collections import OrderedDict
import numpy as np
from holoviews import Store
from holoviews.element import RGB, Image
from holoviews.element.comparison import ComparisonTestCase
def test_unindexed_quadmesh(self):
    plot = self.ds_unindexed.hvplot.quadmesh(x='lon', y='lat')
    assert len(plot.kdims) == 2
    assert plot.kdims[0].name == 'time'
    assert plot.kdims[1].name == 'nsamples'
    p = plot[1, 0]
    assert len(p.kdims) == 2
    assert p.kdims[0].name == 'lon'
    assert p.kdims[1].name == 'lat'