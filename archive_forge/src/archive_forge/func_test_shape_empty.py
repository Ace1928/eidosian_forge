import math
import numpy as np
import pytest
from shapely import (
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry, EmptyGeometry
def test_shape_empty():
    empty_mp = MultiPolygon()
    empty_json = mapping(empty_mp)
    empty_shape = shape(empty_json)
    assert empty_shape.is_empty