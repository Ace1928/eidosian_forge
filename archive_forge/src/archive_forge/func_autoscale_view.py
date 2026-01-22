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
def autoscale_view(self, tight=None, scalex=True, scaley=True, scalez=True):
    """
        Autoscale the view limits using the data limits.

        See `.Axes.autoscale_view` for full documentation.  Because this
        function applies to 3D Axes, it also takes a *scalez* argument.
        """
    if tight is None:
        _tight = self._tight
        if not _tight:
            for artist in self._children:
                if isinstance(artist, mimage.AxesImage):
                    _tight = True
                elif isinstance(artist, (mlines.Line2D, mpatches.Patch)):
                    _tight = False
                    break
    else:
        _tight = self._tight = bool(tight)
    if scalex and self.get_autoscalex_on():
        x0, x1 = self.xy_dataLim.intervalx
        xlocator = self.xaxis.get_major_locator()
        x0, x1 = xlocator.nonsingular(x0, x1)
        if self._xmargin > 0:
            delta = (x1 - x0) * self._xmargin
            x0 -= delta
            x1 += delta
        if not _tight:
            x0, x1 = xlocator.view_limits(x0, x1)
        self.set_xbound(x0, x1)
    if scaley and self.get_autoscaley_on():
        y0, y1 = self.xy_dataLim.intervaly
        ylocator = self.yaxis.get_major_locator()
        y0, y1 = ylocator.nonsingular(y0, y1)
        if self._ymargin > 0:
            delta = (y1 - y0) * self._ymargin
            y0 -= delta
            y1 += delta
        if not _tight:
            y0, y1 = ylocator.view_limits(y0, y1)
        self.set_ybound(y0, y1)
    if scalez and self.get_autoscalez_on():
        z0, z1 = self.zz_dataLim.intervalx
        zlocator = self.zaxis.get_major_locator()
        z0, z1 = zlocator.nonsingular(z0, z1)
        if self._zmargin > 0:
            delta = (z1 - z0) * self._zmargin
            z0 -= delta
            z1 += delta
        if not _tight:
            z0, z1 = zlocator.view_limits(z0, z1)
        self.set_zbound(z0, z1)