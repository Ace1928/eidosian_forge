import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_idxmapping_key_len_check(self):
    try:
        MultiDimensionalMapping(initial_items=self.init_item_odict)
        raise AssertionError('Invalid key length check failed.')
    except KeyError:
        pass