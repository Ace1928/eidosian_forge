import os
import tempfile
from unittest import SkipTest
from collections import OrderedDict
import numpy as np
from holoviews import Store
from holoviews.element import RGB, Image
from holoviews.element.comparison import ComparisonTestCase
def test_dataset_infer_dimension_params_from_xarray_attrs(self):
    ds = self.xarr_with_attrs.hvplot.dataset()
    self.assertEqual(ds.kdims[0].label, 'Declination')
    self.assertEqual(ds.kdims[1].label, 'Right Ascension')
    self.assertEqual(ds.vdims[0].label, 'luminosity')
    self.assertEqual(ds.vdims[0].unit, 'lm')