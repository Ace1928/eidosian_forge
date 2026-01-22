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
@image_comparison(['default_edges.png'], remove_text=True, style='default')
def test_default_edges():
    plt.rcParams['text.kerning_factor'] = 6
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2)
    ax1.plot(np.arange(10), np.arange(10), 'x', np.arange(10) + 1, np.arange(10), 'o')
    ax2.bar(np.arange(10), np.arange(10), align='edge')
    ax3.text(0, 0, 'BOX', size=24, bbox=dict(boxstyle='sawtooth'))
    ax3.set_xlim((-1, 1))
    ax3.set_ylim((-1, 1))
    pp1 = mpatches.PathPatch(mpath.Path([(0, 0), (1, 0), (1, 1), (0, 0)], [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3, mpath.Path.CLOSEPOLY]), fc='none', transform=ax4.transData)
    ax4.add_patch(pp1)