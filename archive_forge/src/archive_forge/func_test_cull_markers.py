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
def test_cull_markers():
    x = np.random.random(20000)
    y = np.random.random(20000)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'k.')
    ax.set_xlim(2, 3)
    pdf = io.BytesIO()
    fig.savefig(pdf, format='pdf')
    assert len(pdf.getvalue()) < 8000
    svg = io.BytesIO()
    fig.savefig(svg, format='svg')
    assert len(svg.getvalue()) < 20000