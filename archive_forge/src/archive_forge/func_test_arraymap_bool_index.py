import numpy as np
import pytest
from skimage.util._map_array import map_array, ArrayMap
from skimage._shared import testing
def test_arraymap_bool_index():
    in_values = np.unique(np.random.randint(0, 200, size=5))
    out_values = np.random.random(len(in_values))
    m = ArrayMap(in_values, out_values)
    image = np.random.randint(1, len(in_values), size=(512, 512))
    assert np.all(m[image] < 1)
    positive = np.ones(len(m), dtype=bool)
    positive[0] = False
    m[positive] += 1
    assert np.all(m[image] >= 1)