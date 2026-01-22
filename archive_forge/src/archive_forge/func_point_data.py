import datetime as dt
from unittest import SkipTest, skipIf
import colorcet as cc
import numpy as np
import pandas as pd
import pytest
from numpy import nan
from packaging.version import Version
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import apply_when
from holoviews.streams import Tap
from holoviews.util import render
import logging
@pytest.fixture()
def point_data():
    num = 100
    np.random.seed(1)
    dists = {cat: pd.DataFrame({'x': np.random.normal(x, s, num), 'y': np.random.normal(y, s, num), 's': s, 'val': val, 'cat': cat}) for x, y, s, val, cat in [(2, 2, 0.03, 0, 'd1'), (2, -2, 0.1, 1, 'd2'), (-2, -2, 0.5, 2, 'd3'), (-2, 2, 1.0, 3, 'd4'), (0, 0, 3.0, 4, 'd5')]}
    df = pd.concat(dists, ignore_index=True)
    return df