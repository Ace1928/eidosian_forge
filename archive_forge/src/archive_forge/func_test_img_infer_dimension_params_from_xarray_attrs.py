import os
import tempfile
from unittest import SkipTest
from collections import OrderedDict
import numpy as np
from holoviews import Store
from holoviews.element import RGB, Image
from holoviews.element.comparison import ComparisonTestCase
def test_img_infer_dimension_params_from_xarray_attrs(self):
    img = self.xarr_with_attrs.hvplot.image(clim=(0, 2))
    self.assertEqual(img.kdims[0].label, 'Declination')
    self.assertEqual(img.kdims[1].label, 'Right Ascension')
    self.assertEqual(img.vdims[0].label, 'luminosity')
    self.assertEqual(img.vdims[0].unit, 'lm')
    self.assertEqual(img.vdims[0].range, (0, 2))