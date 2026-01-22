import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_interpolate_curve_mid_with_values(self):
    interpolated = interpolate_curve(Curve([(0, 0, 'A'), (1, 0.5, 'B'), (2, 1, 'C')], vdims=['y', 'z']), interpolation='steps-mid')
    curve = Curve([(0, 0, 'A'), (0.5, 0, 'A'), (0.5, 0.5, 'B'), (1.5, 0.5, 'B'), (1.5, 1, 'C'), (2, 1, 'C')], vdims=['y', 'z'])
    self.assertEqual(interpolated, curve)