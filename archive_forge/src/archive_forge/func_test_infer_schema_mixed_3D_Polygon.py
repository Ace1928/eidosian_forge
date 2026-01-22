from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_mixed_3D_Polygon():
    df = GeoDataFrame(geometry=[city_hall_boundaries, polygon_3D])
    assert infer_schema(df) == {'geometry': ['3D Polygon', 'Polygon'], 'properties': OrderedDict()}