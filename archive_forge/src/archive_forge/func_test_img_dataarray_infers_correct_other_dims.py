import os
import tempfile
from unittest import SkipTest
from collections import OrderedDict
import numpy as np
from holoviews import Store
from holoviews.element import RGB, Image
from holoviews.element.comparison import ComparisonTestCase
def test_img_dataarray_infers_correct_other_dims(self):
    img = self.da_img_by_time[0].hvplot()
    self.assertEqual(img, Image(self.da_img_by_time[0], ['lon', 'lat'], ['value']))