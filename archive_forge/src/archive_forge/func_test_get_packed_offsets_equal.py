from collections import namedtuple
import io
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.offsetbox import (
@pytest.mark.parametrize('widths, total, sep, expected', [_Params([3, 2, 1], total=6, sep=None, expected=(6, [0, 2, 4])), _Params([3, 2, 1, 0.5], total=2, sep=None, expected=(2, [0, 0.5, 1, 1.5])), _Params([0.5, 1, 0.2], total=None, sep=1, expected=(6, [0, 2, 4]))])
def test_get_packed_offsets_equal(widths, total, sep, expected):
    result = _get_packed_offsets(widths, total, sep, mode='equal')
    assert result[0] == expected[0]
    assert_allclose(result[1], expected[1])