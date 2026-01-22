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
def mouse_init(self, rotate_btn=1, pan_btn=2, zoom_btn=3):
    """
        Set the mouse buttons for 3D rotation and zooming.

        Parameters
        ----------
        rotate_btn : int or list of int, default: 1
            The mouse button or buttons to use for 3D rotation of the axes.
        pan_btn : int or list of int, default: 2
            The mouse button or buttons to use to pan the 3D axes.
        zoom_btn : int or list of int, default: 3
            The mouse button or buttons to use to zoom the 3D axes.
        """
    self.button_pressed = None
    self._rotate_btn = np.atleast_1d(rotate_btn).tolist()
    self._pan_btn = np.atleast_1d(pan_btn).tolist()
    self._zoom_btn = np.atleast_1d(zoom_btn).tolist()