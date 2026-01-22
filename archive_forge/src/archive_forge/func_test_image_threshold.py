import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_image_threshold(self):
    img = Image(np.array([[0, 1, 0], [3, 4, 5.0]]))
    op_img = threshold(img)
    self.assertEqual(op_img, img.clone(np.array([[0, 1, 0], [1, 1, 1]]), group='Threshold'))