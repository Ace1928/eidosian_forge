import io
from itertools import chain
import numpy as np
import pytest
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.artist as martist
import matplotlib.backend_bases as mbackend_bases
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@check_figures_equal(extensions=['png'])
def test_clipping_zoom(fig_test, fig_ref):
    ax_test = fig_test.add_axes([0, 0, 1, 1])
    l, = ax_test.plot([-3, 3], [-3, 3])
    p = mpath.Path([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    p = mpatches.PathPatch(p, transform=ax_test.transData)
    l.set_clip_path(p)
    ax_ref = fig_ref.add_axes([0, 0, 1, 1])
    ax_ref.plot([-3, 3], [-3, 3])
    ax_ref.set(xlim=(0.5, 0.75), ylim=(0.5, 0.75))
    ax_test.set(xlim=(0.5, 0.75), ylim=(0.5, 0.75))