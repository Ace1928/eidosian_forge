from collections import OrderedDict
from shapely.geometry import (
import pandas as pd
import pytest
import numpy as np
from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema
def test_infer_schema_only_3D_linestrings():
    df = GeoDataFrame(geometry=[linestring_3D, linestring_3D])
    assert infer_schema(df) == {'geometry': '3D LineString', 'properties': OrderedDict()}