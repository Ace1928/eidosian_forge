import numpy as np
import param
import pytest
from packaging.version import Version
from holoviews import Annotation, Arrow, HLine, Spline, Text, VLine
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
def test_spline_clone(self):
    points = [(-0.3, -0.3), (0, 0), (0.25, -0.25), (0.3, 0.3)]
    spline = Spline((points, [])).clone()
    self.assertEqual(spline.dimension_values(0), np.array([-0.3, 0, 0.25, 0.3]))
    self.assertEqual(spline.dimension_values(1), np.array([-0.3, 0, -0.25, 0.3]))