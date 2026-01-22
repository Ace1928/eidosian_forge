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
@_preprocess_data()
def bar3d(self, x, y, z, dx, dy, dz, color=None, zsort='average', shade=True, lightsource=None, *args, **kwargs):
    """
        Generate a 3D barplot.

        This method creates three-dimensional barplot where the width,
        depth, height, and color of the bars can all be uniquely set.

        Parameters
        ----------
        x, y, z : array-like
            The coordinates of the anchor point of the bars.

        dx, dy, dz : float or array-like
            The width, depth, and height of the bars, respectively.

        color : sequence of colors, optional
            The color of the bars can be specified globally or
            individually. This parameter can be:

            - A single color, to color all bars the same color.
            - An array of colors of length N bars, to color each bar
              independently.
            - An array of colors of length 6, to color the faces of the
              bars similarly.
            - An array of colors of length 6 * N bars, to color each face
              independently.

            When coloring the faces of the boxes specifically, this is
            the order of the coloring:

            1. -Z (bottom of box)
            2. +Z (top of box)
            3. -Y
            4. +Y
            5. -X
            6. +X

        zsort : str, optional
            The z-axis sorting scheme passed onto `~.art3d.Poly3DCollection`

        shade : bool, default: True
            When true, this shades the dark sides of the bars (relative
            to the plot's source of light).

        lightsource : `~matplotlib.colors.LightSource`
            The lightsource to use when *shade* is True.

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Any additional keyword arguments are passed onto
            `~.art3d.Poly3DCollection`.

        Returns
        -------
        collection : `~.art3d.Poly3DCollection`
            A collection of three-dimensional polygons representing the bars.
        """
    had_data = self.has_data()
    x, y, z, dx, dy, dz = np.broadcast_arrays(np.atleast_1d(x), y, z, dx, dy, dz)
    minx = np.min(x)
    maxx = np.max(x + dx)
    miny = np.min(y)
    maxy = np.max(y + dy)
    minz = np.min(z)
    maxz = np.max(z + dz)
    cuboid = np.array([((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))])
    polys = np.empty(x.shape + cuboid.shape)
    for i, p, dp in [(0, x, dx), (1, y, dy), (2, z, dz)]:
        p = p[..., np.newaxis, np.newaxis]
        dp = dp[..., np.newaxis, np.newaxis]
        polys[..., i] = p + dp * cuboid[..., i]
    polys = polys.reshape((-1,) + polys.shape[2:])
    facecolors = []
    if color is None:
        color = [self._get_patches_for_fill.get_next_color()]
    color = list(mcolors.to_rgba_array(color))
    if len(color) == len(x):
        for c in color:
            facecolors.extend([c] * 6)
    else:
        facecolors = color
        if len(facecolors) < len(x):
            facecolors *= 6 * len(x)
    col = art3d.Poly3DCollection(polys, *args, zsort=zsort, facecolors=facecolors, shade=shade, lightsource=lightsource, **kwargs)
    self.add_collection(col)
    self.auto_scale_xyz((minx, maxx), (miny, maxy), (minz, maxz), had_data)
    return col