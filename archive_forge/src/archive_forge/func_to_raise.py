import os
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
import pytest
from .test_file import FIONA_MARK, PYOGRIO_MARK
def to_raise(self, error_type, error_match):
    _expected_exceptions[self.composite_key] = _ExpectedError(error_type, error_match)