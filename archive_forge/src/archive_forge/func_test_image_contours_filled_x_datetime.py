import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_image_contours_filled_x_datetime(self):
    x = np.array(['2023-09-01', '2023-09-05', '2023-09-09'], dtype='datetime64')
    y = np.array([6, 7])
    z = np.array([[0, 2, 0], [0, 2, 0]])
    img = Image((x, y, z))
    msg = 'Datetime spatial coordinates are not supported for filled contour calculations.'
    with pytest.raises(RuntimeError, match=msg):
        _ = contours(img, filled=True, levels=[0.5, 1.5])