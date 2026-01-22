import base64
import io
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from matplotlib.testing.decorators import (
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.path import Path
@pytest.mark.parametrize('angle', [0, np.pi / 4, np.pi / 3, np.pi / 2])
@pytest.mark.parametrize('offset', [0, 0.5])
def test_angled_antiparallel(angle, offset):
    scale = 5
    np.random.seed(19680801)
    vert_offsets = (np.random.rand(15) - offset) * scale
    vert_offsets[0] = 0
    vert_offsets[1] = 1
    x = np.sin(angle) * vert_offsets
    y = np.cos(angle) * vert_offsets
    x_max = x[1:].max()
    x_min = x[1:].min()
    y_max = y[1:].max()
    y_min = y[1:].min()
    if offset > 0:
        p_expected = Path([[0, 0], [x_max, y_max], [x_min, y_min], [x[-1], y[-1]], [0, 0]], codes=[1, 2, 2, 2, 0])
    else:
        p_expected = Path([[0, 0], [x_max, y_max], [x[-1], y[-1]], [0, 0]], codes=[1, 2, 2, 0])
    p = Path(np.vstack([x, y]).T)
    p2 = p.cleaned(simplify=True)
    assert_array_almost_equal(p_expected.vertices, p2.vertices)
    assert_array_equal(p_expected.codes, p2.codes)