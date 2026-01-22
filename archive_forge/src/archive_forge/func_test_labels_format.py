import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
def test_labels_format(self):
    plot = self.df.hvplot('x', 'y', text='({x}, {y})', kind='labels')
    assert list(plot.dimensions()) == [Dimension('x'), Dimension('y'), Dimension('label')]
    assert list(plot.data['label']) == ['(1, 2)', '(3, 4)', '(5, 6)']