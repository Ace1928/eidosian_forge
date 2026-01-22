import numpy as np
from holoviews import Curve, HoloMap, Image, Overlay
from holoviews.core.options import Store, StoreOptions
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl  # noqa Register backend
def test_partitioned_format(self):
    out = StoreOptions.merge_options(['plot', 'style'], plot={'Image': dict(fig_size=150)}, style={'Image': dict(cmap='Blues')})
    self.assertEqual(out, self.expected)