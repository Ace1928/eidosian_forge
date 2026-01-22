import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_image_casting(self):
    img = Image([], bounds=2)
    self.assertEqual(img, Image(img))