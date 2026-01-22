import itertools
import numpy as np
from holoviews.core import Collator, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_collate_layout_hmap(self):
    layout = self.nested_overlay + self.nested_overlay
    collated = Collator(kdims=['delta'], merge_type=NdOverlay)
    for k, v in self.nesting_hmap.groupby(['delta']).items():
        collated[k] = v + v
    collated = collated()
    self.assertEqual(repr(collated), repr(layout))
    self.assertEqual(collated.dimensions(), layout.dimensions())