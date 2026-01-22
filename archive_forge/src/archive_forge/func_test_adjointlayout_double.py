import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
def test_adjointlayout_double(self):
    layout = self.view1 << self.view2
    self.assertEqual(layout.main, self.view1)
    self.assertEqual(layout.right, self.view2)