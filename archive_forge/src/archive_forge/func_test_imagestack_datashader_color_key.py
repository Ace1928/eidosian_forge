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
def test_imagestack_datashader_color_key():
    d = np.arange(23)
    df = pd.DataFrame({'x': d, 'y': d, 'language': list(map(str, d))})
    points = Points(df, ['x', 'y'], ['language'])
    op = datashade(points, aggregator=ds.by('language', ds.count()), color_key=cc.glasbey_light)
    render(op)