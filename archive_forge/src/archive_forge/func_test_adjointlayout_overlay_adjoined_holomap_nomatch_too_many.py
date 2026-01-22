import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
def test_adjointlayout_overlay_adjoined_holomap_nomatch_too_many(self):
    dim_view = self.view3.clone(kdims=['x', 'y'])
    with self.assertRaises(ValueError):
        (self.view1 << self.view2 << self.view3) * (self.hmap << dim_view)