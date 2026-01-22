import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_interpolate_curve_post(self):
    interpolated = interpolate_curve(Curve([0, 0.5, 1]), interpolation='steps-post')
    curve = Curve([(0, 0), (1, 0), (1, 0.5), (2, 0.5), (2, 1)])
    self.assertEqual(interpolated, curve)