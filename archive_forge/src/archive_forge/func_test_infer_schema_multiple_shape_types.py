from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_multiple_shape_types():
    df = GeoDataFrame(geometry=[MultiPolygon((city_hall_boundaries, vauquelin_place)), city_hall_boundaries, MultiLineString(city_hall_walls), city_hall_walls[0], MultiPoint([city_hall_entrance, city_hall_balcony]), city_hall_balcony])
    assert infer_schema(df) == {'geometry': ['MultiPolygon', 'Polygon', 'MultiLineString', 'LineString', 'MultiPoint', 'Point'], 'properties': OrderedDict()}