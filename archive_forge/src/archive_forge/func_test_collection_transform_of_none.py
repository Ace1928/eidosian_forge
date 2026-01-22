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
def test_collection_transform_of_none():
    ax = plt.axes()
    ax.set_xlim(1, 3)
    ax.set_ylim(1, 3)
    xy_data = (2, 2)
    xy_pix = ax.transData.transform(xy_data)
    e = mpatches.Ellipse(xy_data, width=1, height=1)
    c = mcollections.PatchCollection([e], facecolor='yellow', alpha=0.5)
    ax.add_collection(c)
    assert c.get_offset_transform() + c.get_transform() == ax.transData
    e = mpatches.Ellipse(xy_pix, width=120, height=120)
    c = mcollections.PatchCollection([e], facecolor='coral', alpha=0.5)
    c.set_transform(None)
    ax.add_collection(c)
    assert isinstance(c.get_transform(), mtransforms.IdentityTransform)
    e = mpatches.Ellipse(xy_pix, width=100, height=100)
    c = mcollections.PatchCollection([e], transform=mtransforms.IdentityTransform(), alpha=0.5)
    ax.add_collection(c)
    assert isinstance(c.get_offset_transform(), mtransforms.IdentityTransform)