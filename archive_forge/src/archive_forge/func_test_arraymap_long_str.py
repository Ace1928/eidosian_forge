import numpy as np
import pytest
from skimage.util._map_array import map_array, ArrayMap
from skimage._shared import testing
def test_arraymap_long_str():
    labels = np.random.randint(0, 40, size=(24, 25))
    in_values = np.unique(labels)
    out_values = np.random.random(in_values.shape)
    m = ArrayMap(in_values, out_values)
    assert len(str(m).split('\n')) == m._max_str_lines + 2