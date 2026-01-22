import time
import uuid
from collections import deque
import numpy as np
import param
import pytest
from holoviews import Dimension, GridSpace, Layout, NdLayout, NdOverlay
from holoviews.core.options import Store
from holoviews.core.spaces import Callable, DynamicMap, HoloMap
from holoviews.element import Curve, Image, Points, Scatter, Text
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import histogram
from holoviews.plotting.util import initialize_dynamic
from holoviews.streams import (
from holoviews.util import Dynamic
from ..utils import LoggingComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement
def test_dynamic_groupby_kdims_and_streams(self):

    def plot_function(mydim, data):
        return Scatter(data[data[:, 2] == mydim])
    buff = Buffer(data=np.empty((0, 3)))
    dmap = DynamicMap(plot_function, streams=[buff], kdims='mydim').redim.values(mydim=[0, 1, 2])
    ndlayout = dmap.groupby('mydim', container_type=NdLayout)
    self.assertIsInstance(ndlayout[0], DynamicMap)
    data = np.array([(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    buff.send(data)
    self.assertIs(ndlayout[0].callback.inputs[0], dmap)
    self.assertIs(ndlayout[1].callback.inputs[0], dmap)
    self.assertIs(ndlayout[2].callback.inputs[0], dmap)
    self.assertEqual(ndlayout[0][()], Scatter([(0, 0)]))
    self.assertEqual(ndlayout[1][()], Scatter([(1, 1)]))
    self.assertEqual(ndlayout[2][()], Scatter([(2, 2)]))