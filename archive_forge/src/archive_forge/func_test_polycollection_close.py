from datetime import datetime
import io
import itertools
import re
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
from matplotlib.collections import (Collection, LineCollection,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@image_comparison(['polycollection_close.png'], remove_text=True, style='mpl20')
def test_polycollection_close():
    from mpl_toolkits.mplot3d import Axes3D
    vertsQuad = [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [2.0, 3.0], [2.0, 2.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 3.0], [4.0, 1.0], [3.0, 1.0]], [[3.0, 0.0], [3.0, 1.0], [4.0, 1.0], [4.0, 0.0]]]
    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig))
    colors = ['r', 'g', 'b', 'y', 'k']
    zpos = list(range(5))
    poly = mcollections.PolyCollection(vertsQuad * len(zpos), linewidth=0.25)
    poly.set_alpha(0.7)
    zs = []
    cs = []
    for z, c in zip(zpos, colors):
        zs.extend([z] * len(vertsQuad))
        cs.extend([c] * len(vertsQuad))
    poly.set_color(cs)
    ax.add_collection3d(poly, zs=zs, zdir='y')
    ax.set_xlim3d(0, 4)
    ax.set_zlim3d(0, 3)
    ax.set_ylim3d(0, 4)