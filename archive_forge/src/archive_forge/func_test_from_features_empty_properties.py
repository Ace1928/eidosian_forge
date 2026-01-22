import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import Point, Polygon
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, points_from_xy, read_file
from geopandas.array import GeometryArray, GeometryDtype, from_shapely
from geopandas._compat import ignore_shapely2_warnings
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytest
def test_from_features_empty_properties(self):
    geojson_properties_object = '{\n          "type": "FeatureCollection",\n          "features": [\n            {\n              "type": "Feature",\n              "properties": {},\n              "geometry": {\n                "type": "Polygon",\n                "coordinates": [\n                  [\n                    [\n                      11.3456529378891,\n                      46.49461446367692\n                    ],\n                    [\n                      11.345674395561216,\n                      46.494097442978195\n                    ],\n                    [\n                      11.346918940544128,\n                      46.49385370294394\n                    ],\n                    [\n                      11.347616314888,\n                      46.4938352377453\n                    ],\n                    [\n                      11.347514390945435,\n                      46.49466985846028\n                    ],\n                    [\n                      11.3456529378891,\n                      46.49461446367692\n                    ]\n                  ]\n                ]\n              }\n            }\n          ]\n        }'
    geojson_properties_null = '{\n          "type": "FeatureCollection",\n          "features": [\n            {\n              "type": "Feature",\n              "properties": null,\n              "geometry": {\n                "type": "Polygon",\n                "coordinates": [\n                  [\n                    [\n                      11.3456529378891,\n                      46.49461446367692\n                    ],\n                    [\n                      11.345674395561216,\n                      46.494097442978195\n                    ],\n                    [\n                      11.346918940544128,\n                      46.49385370294394\n                    ],\n                    [\n                      11.347616314888,\n                      46.4938352377453\n                    ],\n                    [\n                      11.347514390945435,\n                      46.49466985846028\n                    ],\n                    [\n                      11.3456529378891,\n                      46.49461446367692\n                    ]\n                  ]\n                ]\n              }\n            }\n          ]\n        }'
    gjson_po = json.loads(geojson_properties_object)
    gdf1 = GeoDataFrame.from_features(gjson_po)
    gjson_null = json.loads(geojson_properties_null)
    gdf2 = GeoDataFrame.from_features(gjson_null)
    assert_frame_equal(gdf1, gdf2)