import math
import numpy as np
import pytest
from shapely import (
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry, EmptyGeometry
def test_empty_linear_ring(self):
    assert LinearRing().is_empty
    assert LinearRing(None).is_empty
    assert LinearRing([]).is_empty
    assert LinearRing(empty_generator()).is_empty