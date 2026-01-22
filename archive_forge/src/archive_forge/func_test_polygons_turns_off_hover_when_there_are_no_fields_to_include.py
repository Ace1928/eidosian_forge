import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_polygons_turns_off_hover_when_there_are_no_fields_to_include(self):
    polygons = self.polygons.hvplot(geo=True)
    opts = hv.Store.lookup_options('bokeh', polygons, 'plot').kwargs
    assert 'hover' not in opts.get('tools')