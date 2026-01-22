import math
from typing import Sequence
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon, GeometryCollection
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, sjoin, sjoin_nearest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
def test_geometry_name(self):
    polydf_original_geom_name = self.polydf.geometry.name
    self.polydf = self.polydf.rename(columns={'geometry': 'new_geom'}).set_geometry('new_geom')
    assert polydf_original_geom_name != self.polydf.geometry.name
    res = sjoin(self.polydf, self.pointdf, how='left')
    assert self.polydf.geometry.name == res.geometry.name