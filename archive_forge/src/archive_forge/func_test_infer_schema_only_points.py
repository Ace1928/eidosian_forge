from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_only_points():
    df = GeoDataFrame(geometry=[city_hall_entrance, city_hall_balcony])
    assert infer_schema(df) == {'geometry': 'Point', 'properties': OrderedDict()}