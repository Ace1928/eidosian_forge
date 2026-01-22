import numpy as np
import param
import pytest
from packaging.version import Version
from holoviews import Annotation, Arrow, HLine, Spline, Text, VLine
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
def test_hline_dimension_values(self):
    hline = HLine(0)
    self.assertTrue(all((not np.isfinite(v) for v in hline.range(0))))
    self.assertEqual(hline.range(1), (0, 0))
    hline = HLine(np.array([0]))
    self.assertTrue(all((not np.isfinite(v) for v in hline.range(0))))
    self.assertEqual(hline.range(1), (0, 0))
    hline = HLine(np.array(0))
    self.assertTrue(all((not np.isfinite(v) for v in hline.range(0))))
    self.assertEqual(hline.range(1), (0, 0))