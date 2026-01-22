from collections import defaultdict
import functools
import itertools
import math
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation
from . import art3d
from . import proj3d
from . import axis3d
def shareview(self, other):
    """
        Share the view angles with *other*.

        This is equivalent to passing ``shareview=other`` when
        constructing the Axes, and cannot be used if the view angles are
        already being shared with another Axes.
        """
    _api.check_isinstance(Axes3D, other=other)
    if self._shareview is not None and other is not self._shareview:
        raise ValueError('view angles are already shared')
    self._shared_axes['view'].join(self, other)
    self._shareview = other
    vertical_axis = {0: 'x', 1: 'y', 2: 'z'}[other._vertical_axis]
    self.view_init(elev=other.elev, azim=other.azim, roll=other.roll, vertical_axis=vertical_axis, share=True)