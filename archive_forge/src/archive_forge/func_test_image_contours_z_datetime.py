import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_image_contours_z_datetime(self):
    z = np.array([['2023-09-10', '2023-09-10'], ['2023-09-10', '2023-09-12']], dtype='datetime64')
    img = Image(z)
    op_contours = contours(img, levels=[np.datetime64('2023-09-11')])
    np.testing.assert_array_almost_equal(op_contours.dimension_values('x'), [0.25, 0.0])
    np.testing.assert_array_almost_equal(op_contours.dimension_values('y'), [0.0, -0.25])
    expected_z = np.array([dt.datetime(2023, 9, 11, 0, 0, tzinfo=dt.timezone.utc), dt.datetime(2023, 9, 11, 0, 0, tzinfo=dt.timezone.utc)], dtype=object)
    np.testing.assert_array_equal(op_contours.dimension_values('z'), expected_z)