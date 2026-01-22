import itertools
from packaging.version import Version
import warnings
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.affinity import rotate
from shapely.geometry import (
from geopandas import GeoDataFrame, GeoSeries, read_file
from geopandas.datasets import get_path
import geopandas._compat as compat
from geopandas.plotting import GeoplotAccessor
import pytest
import matplotlib.pyplot as plt
def test_polygon_patch():
    from geopandas.plotting import _PolygonPatch
    from matplotlib.patches import PathPatch
    polygon = Point(0, 0).buffer(10.0).difference(MultiPoint([(-5, 0), (5, 0)]).buffer(3.0))
    patch = _PolygonPatch(polygon)
    assert isinstance(patch, PathPatch)
    path = patch.get_path()
    if compat.GEOS_GE_390:
        assert len(path.vertices) == len(path.codes) == 195
    else:
        assert len(path.vertices) == len(path.codes) == 198