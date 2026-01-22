import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_image_contours_filled_empty(self):
    img = Image(np.array([[0, 1, 0], [3, 4, 5.0], [6, 7, 8]]))
    op_contours = contours(img, filled=True, levels=[20.0, 23.0])
    polys = Polygons([], vdims=img.vdims[0].clone(range=(20.0, 23.0)))
    self.assertEqual(op_contours, polys)