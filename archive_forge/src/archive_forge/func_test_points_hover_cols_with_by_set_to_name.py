import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_points_hover_cols_with_by_set_to_name(self):
    points = self.cities.hvplot(by='name')
    assert isinstance(points, hv.core.overlay.NdOverlay)
    assert points.kdims == ['name']
    assert points.vdims == []
    for element in points.values():
        assert element.kdims == ['x', 'y']
        assert element.vdims == []