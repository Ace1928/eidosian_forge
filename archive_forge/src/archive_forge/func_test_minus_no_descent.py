from tempfile import TemporaryFile
import numpy as np
from packaging.version import parse as parse_version
import pytest
import matplotlib as mpl
from matplotlib import dviread
from matplotlib.testing import _has_tex_package
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
import matplotlib.pyplot as plt
@pytest.mark.parametrize('fontsize', [8, 10, 12])
def test_minus_no_descent(fontsize):
    mpl.style.use('mpl20')
    mpl.rcParams['font.size'] = fontsize
    heights = {}
    fig = plt.figure()
    for vals in [(1,), (-1,), (-1, 1)]:
        fig.clear()
        for x in vals:
            fig.text(0.5, 0.5, f'${x}$', usetex=True)
        fig.canvas.draw()
        heights[vals] = (np.array(fig.canvas.buffer_rgba())[..., 0] != 255).any(axis=1).sum()
    assert len({*heights.values()}) == 1