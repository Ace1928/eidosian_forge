import os
import tempfile
from unittest import SkipTest
from collections import OrderedDict
import numpy as np
from holoviews import Store
from holoviews.element import RGB, Image
from holoviews.element.comparison import ComparisonTestCase
def test_line_infer_dimension_params_from_xarray_attrs(self):
    hmap = self.xarr_with_attrs.hvplot.line(groupby='x', dynamic=False)
    self.assertEqual(hmap.kdims[0].label, 'Declination')
    self.assertEqual(hmap.last.kdims[0].label, 'Right Ascension')
    self.assertEqual(hmap.last.vdims[0].label, 'luminosity')
    self.assertEqual(hmap.last.vdims[0].unit, 'lm')