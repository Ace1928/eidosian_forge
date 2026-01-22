import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_image_contours_y_datetime(self):
    x = [14, 15, 16]
    y = np.array(['2023-09-01', '2023-09-03'], dtype='datetime64')
    z = np.array([[0, 1, 0], [0, 1, 0]])
    img = Image((x, y, z))
    op_contours = contours(img, levels=[0.5])
    np.testing.assert_array_almost_equal(op_contours.dimension_values('x').astype(float), [14.5, 14.5, np.nan, 15.5, 15.5])
    tz = dt.timezone.utc
    expected_y = np.array([dt.datetime(2023, 9, 3, tzinfo=tz), dt.datetime(2023, 9, 1, tzinfo=tz), np.nan, dt.datetime(2023, 9, 1, tzinfo=tz), dt.datetime(2023, 9, 3, tzinfo=tz)], dtype=object)
    y = op_contours.dimension_values('y')
    mask = np.array([True, True, False, True, True])
    np.testing.assert_array_equal(y[mask], expected_y[mask])
    np.testing.assert_array_equal(y[~mask].astype(float), expected_y[~mask].astype(float))
    np.testing.assert_array_almost_equal(op_contours.dimension_values('z'), [0.5] * 5)