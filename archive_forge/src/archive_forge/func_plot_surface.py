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
def plot_surface(self, X, Y, Z, *, norm=None, vmin=None, vmax=None, lightsource=None, **kwargs):
    """
        Create a surface plot.

        By default, it will be colored in shades of a solid color, but it also
        supports colormapping by supplying the *cmap* argument.

        .. note::

           The *rcount* and *ccount* kwargs, which both default to 50,
           determine the maximum number of samples used in each direction.  If
           the input data is larger, it will be downsampled (by slicing) to
           these numbers of points.

        .. note::

           To maximize rendering speed consider setting *rstride* and *cstride*
           to divisors of the number of rows minus 1 and columns minus 1
           respectively. For example, given 51 rows rstride can be any of the
           divisors of 50.

           Similarly, a setting of *rstride* and *cstride* equal to 1 (or
           *rcount* and *ccount* equal the number of rows and columns) can use
           the optimized path.

        Parameters
        ----------
        X, Y, Z : 2D arrays
            Data values.

        rcount, ccount : int
            Maximum number of samples used in each direction.  If the input
            data is larger, it will be downsampled (by slicing) to these
            numbers of points.  Defaults to 50.

        rstride, cstride : int
            Downsampling stride in each direction.  These arguments are
            mutually exclusive with *rcount* and *ccount*.  If only one of
            *rstride* or *cstride* is set, the other defaults to 10.

            'classic' mode uses a default of ``rstride = cstride = 10`` instead
            of the new default of ``rcount = ccount = 50``.

        color : color-like
            Color of the surface patches.

        cmap : Colormap
            Colormap of the surface patches.

        facecolors : array-like of colors.
            Colors of each individual patch.

        norm : Normalize
            Normalization for the colormap.

        vmin, vmax : float
            Bounds for the normalization.

        shade : bool, default: True
            Whether to shade the facecolors.  Shading is always disabled when
            *cmap* is specified.

        lightsource : `~matplotlib.colors.LightSource`
            The lightsource to use when *shade* is True.

        **kwargs
            Other keyword arguments are forwarded to `.Poly3DCollection`.
        """
    had_data = self.has_data()
    if Z.ndim != 2:
        raise ValueError('Argument Z must be 2-dimensional.')
    Z = cbook._to_unmasked_float_array(Z)
    X, Y, Z = np.broadcast_arrays(X, Y, Z)
    rows, cols = Z.shape
    has_stride = 'rstride' in kwargs or 'cstride' in kwargs
    has_count = 'rcount' in kwargs or 'ccount' in kwargs
    if has_stride and has_count:
        raise ValueError('Cannot specify both stride and count arguments')
    rstride = kwargs.pop('rstride', 10)
    cstride = kwargs.pop('cstride', 10)
    rcount = kwargs.pop('rcount', 50)
    ccount = kwargs.pop('ccount', 50)
    if mpl.rcParams['_internal.classic_mode']:
        compute_strides = has_count
    else:
        compute_strides = not has_stride
    if compute_strides:
        rstride = int(max(np.ceil(rows / rcount), 1))
        cstride = int(max(np.ceil(cols / ccount), 1))
    fcolors = kwargs.pop('facecolors', None)
    cmap = kwargs.get('cmap', None)
    shade = kwargs.pop('shade', cmap is None)
    if shade is None:
        raise ValueError('shade cannot be None.')
    colset = []
    if (rows - 1) % rstride == 0 and (cols - 1) % cstride == 0 and (fcolors is None):
        polys = np.stack([cbook._array_patch_perimeters(a, rstride, cstride) for a in (X, Y, Z)], axis=-1)
    else:
        row_inds = list(range(0, rows - 1, rstride)) + [rows - 1]
        col_inds = list(range(0, cols - 1, cstride)) + [cols - 1]
        polys = []
        for rs, rs_next in zip(row_inds[:-1], row_inds[1:]):
            for cs, cs_next in zip(col_inds[:-1], col_inds[1:]):
                ps = [cbook._array_perimeter(a[rs:rs_next + 1, cs:cs_next + 1]) for a in (X, Y, Z)]
                ps = np.array(ps).T
                polys.append(ps)
                if fcolors is not None:
                    colset.append(fcolors[rs][cs])
    if not isinstance(polys, np.ndarray) or not np.isfinite(polys).all():
        new_polys = []
        new_colset = []
        for p, col in itertools.zip_longest(polys, colset):
            new_poly = np.array(p)[np.isfinite(p).all(axis=1)]
            if len(new_poly):
                new_polys.append(new_poly)
                new_colset.append(col)
        polys = new_polys
        if fcolors is not None:
            colset = new_colset
    if fcolors is not None:
        polyc = art3d.Poly3DCollection(polys, edgecolors=colset, facecolors=colset, shade=shade, lightsource=lightsource, **kwargs)
    elif cmap:
        polyc = art3d.Poly3DCollection(polys, **kwargs)
        if isinstance(polys, np.ndarray):
            avg_z = polys[..., 2].mean(axis=-1)
        else:
            avg_z = np.array([ps[:, 2].mean() for ps in polys])
        polyc.set_array(avg_z)
        if vmin is not None or vmax is not None:
            polyc.set_clim(vmin, vmax)
        if norm is not None:
            polyc.set_norm(norm)
    else:
        color = kwargs.pop('color', None)
        if color is None:
            color = self._get_lines.get_next_color()
        color = np.array(mcolors.to_rgba(color))
        polyc = art3d.Poly3DCollection(polys, facecolors=color, shade=shade, lightsource=lightsource, **kwargs)
    self.add_collection(polyc)
    self.auto_scale_xyz(X, Y, Z, had_data)
    return polyc