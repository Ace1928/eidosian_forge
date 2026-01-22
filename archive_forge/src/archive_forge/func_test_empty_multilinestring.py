import math
import numpy as np
import pytest
from shapely import (
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry, EmptyGeometry
def test_empty_multilinestring(self):
    assert MultiLineString([]).is_empty