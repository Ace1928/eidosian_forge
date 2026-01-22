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
def plot_wireframe(self, X, Y, Z, **kwargs):
    """
        Plot a 3D wireframe.

        .. note::

           The *rcount* and *ccount* kwargs, which both default to 50,
           determine the maximum number of samples used in each direction.  If
           the input data is larger, it will be downsampled (by slicing) to
           these numbers of points.

        Parameters
        ----------
        X, Y, Z : 2D arrays
            Data values.

        rcount, ccount : int
            Maximum number of samples used in each direction.  If the input
            data is larger, it will be downsampled (by slicing) to these
            numbers of points.  Setting a count to zero causes the data to be
            not sampled in the corresponding direction, producing a 3D line
            plot rather than a wireframe plot.  Defaults to 50.

        rstride, cstride : int
            Downsampling stride in each direction.  These arguments are
            mutually exclusive with *rcount* and *ccount*.  If only one of
            *rstride* or *cstride* is set, the other defaults to 1.  Setting a
            stride to zero causes the data to be not sampled in the
            corresponding direction, producing a 3D line plot rather than a
            wireframe plot.

            'classic' mode uses a default of ``rstride = cstride = 1`` instead
            of the new default of ``rcount = ccount = 50``.

        **kwargs
            Other keyword arguments are forwarded to `.Line3DCollection`.
        """
    had_data = self.has_data()
    if Z.ndim != 2:
        raise ValueError('Argument Z must be 2-dimensional.')
    X, Y, Z = np.broadcast_arrays(X, Y, Z)
    rows, cols = Z.shape
    has_stride = 'rstride' in kwargs or 'cstride' in kwargs
    has_count = 'rcount' in kwargs or 'ccount' in kwargs
    if has_stride and has_count:
        raise ValueError('Cannot specify both stride and count arguments')
    rstride = kwargs.pop('rstride', 1)
    cstride = kwargs.pop('cstride', 1)
    rcount = kwargs.pop('rcount', 50)
    ccount = kwargs.pop('ccount', 50)
    if mpl.rcParams['_internal.classic_mode']:
        if has_count:
            rstride = int(max(np.ceil(rows / rcount), 1)) if rcount else 0
            cstride = int(max(np.ceil(cols / ccount), 1)) if ccount else 0
    elif not has_stride:
        rstride = int(max(np.ceil(rows / rcount), 1)) if rcount else 0
        cstride = int(max(np.ceil(cols / ccount), 1)) if ccount else 0
    tX, tY, tZ = (np.transpose(X), np.transpose(Y), np.transpose(Z))
    if rstride:
        rii = list(range(0, rows, rstride))
        if rows > 0 and rii[-1] != rows - 1:
            rii += [rows - 1]
    else:
        rii = []
    if cstride:
        cii = list(range(0, cols, cstride))
        if cols > 0 and cii[-1] != cols - 1:
            cii += [cols - 1]
    else:
        cii = []
    if rstride == 0 and cstride == 0:
        raise ValueError('Either rstride or cstride must be non zero')
    if Z.size == 0:
        rii = []
        cii = []
    xlines = [X[i] for i in rii]
    ylines = [Y[i] for i in rii]
    zlines = [Z[i] for i in rii]
    txlines = [tX[i] for i in cii]
    tylines = [tY[i] for i in cii]
    tzlines = [tZ[i] for i in cii]
    lines = [list(zip(xl, yl, zl)) for xl, yl, zl in zip(xlines, ylines, zlines)] + [list(zip(xl, yl, zl)) for xl, yl, zl in zip(txlines, tylines, tzlines)]
    linec = art3d.Line3DCollection(lines, **kwargs)
    self.add_collection(linec)
    self.auto_scale_xyz(X, Y, Z, had_data)
    return linec