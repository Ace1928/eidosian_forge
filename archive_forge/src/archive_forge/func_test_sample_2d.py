import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_sample_2d(self):
    xs = ys = np.linspace(0, 6, 50)
    XS, YS = np.meshgrid(xs, ys)
    values = np.sin(XS)
    sampled = Dataset((xs, ys, values), ['x', 'y'], 'z').sample(y=0)
    self.assertEqual(sampled, Curve((xs, values[0]), vdims='z'))