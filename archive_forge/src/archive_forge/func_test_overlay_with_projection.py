import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_overlay_with_projection(self):
    df = pd.DataFrame({'lon': [0, 10], 'lat': [40, 50], 'v': [0, 1]})
    plot1 = df.hvplot.points(x='lon', y='lat', s=200, c='y', geo=True, tiles='CartoLight')
    plot2 = df.hvplot.points(x='lon', y='lat', c='v', geo=True)
    plot = plot1 * plot2
    hv.renderer('bokeh').get_plot(plot)