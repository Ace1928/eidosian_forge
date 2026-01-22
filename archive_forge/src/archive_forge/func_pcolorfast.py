import functools
import itertools
import logging
import math
from numbers import Integral, Number, Real
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.category  # Register category unit converter as side effect.
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates  # noqa # Register date unit converter as side effect.
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.units as munits
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import (
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
@_preprocess_data()
@_docstring.dedent_interpd
def pcolorfast(self, *args, alpha=None, norm=None, cmap=None, vmin=None, vmax=None, **kwargs):
    """
        Create a pseudocolor plot with a non-regular rectangular grid.

        Call signature::

          ax.pcolorfast([X, Y], C, /, **kwargs)

        This method is similar to `~.Axes.pcolor` and `~.Axes.pcolormesh`.
        It's designed to provide the fastest pcolor-type plotting with the
        Agg backend. To achieve this, it uses different algorithms internally
        depending on the complexity of the input grid (regular rectangular,
        non-regular rectangular or arbitrary quadrilateral).

        .. warning::

           This method is experimental. Compared to `~.Axes.pcolor` or
           `~.Axes.pcolormesh` it has some limitations:

           - It supports only flat shading (no outlines)
           - It lacks support for log scaling of the axes.
           - It does not have a pyplot wrapper.

        Parameters
        ----------
        C : array-like
            The image data. Supported array shapes are:

            - (M, N): an image with scalar data.  Color-mapping is controlled
              by *cmap*, *norm*, *vmin*, and *vmax*.
            - (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
            - (M, N, 4): an image with RGBA values (0-1 float or 0-255 int),
              i.e. including transparency.

            The first two dimensions (M, N) define the rows and columns of
            the image.

            This parameter can only be passed positionally.

        X, Y : tuple or array-like, default: ``(0, N)``, ``(0, M)``
            *X* and *Y* are used to specify the coordinates of the
            quadrilaterals. There are different ways to do this:

            - Use tuples ``X=(xmin, xmax)`` and ``Y=(ymin, ymax)`` to define
              a *uniform rectangular grid*.

              The tuples define the outer edges of the grid. All individual
              quadrilaterals will be of the same size. This is the fastest
              version.

            - Use 1D arrays *X*, *Y* to specify a *non-uniform rectangular
              grid*.

              In this case *X* and *Y* have to be monotonic 1D arrays of length
              *N+1* and *M+1*, specifying the x and y boundaries of the cells.

              The speed is intermediate. Note: The grid is checked, and if
              found to be uniform the fast version is used.

            - Use 2D arrays *X*, *Y* if you need an *arbitrary quadrilateral
              grid* (i.e. if the quadrilaterals are not rectangular).

              In this case *X* and *Y* are 2D arrays with shape (M + 1, N + 1),
              specifying the x and y coordinates of the corners of the colored
              quadrilaterals.

              This is the most general, but the slowest to render.  It may
              produce faster and more compact output using ps, pdf, and
              svg backends, however.

            These arguments can only be passed positionally.

        %(cmap_doc)s

            This parameter is ignored if *C* is RGB(A).

        %(norm_doc)s

            This parameter is ignored if *C* is RGB(A).

        %(vmin_vmax_doc)s

            This parameter is ignored if *C* is RGB(A).

        alpha : float, default: None
            The alpha blending value, between 0 (transparent) and 1 (opaque).

        snap : bool, default: False
            Whether to snap the mesh to pixel boundaries.

        Returns
        -------
        `.AxesImage` or `.PcolorImage` or `.QuadMesh`
            The return type depends on the type of grid:

            - `.AxesImage` for a regular rectangular grid.
            - `.PcolorImage` for a non-regular rectangular grid.
            - `.QuadMesh` for a non-rectangular grid.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Supported additional parameters depend on the type of grid.
            See return types of *image* for further description.
        """
    C = args[-1]
    nr, nc = np.shape(C)[:2]
    if len(args) == 1:
        style = 'image'
        x = [0, nc]
        y = [0, nr]
    elif len(args) == 3:
        x, y = args[:2]
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim == 1 and y.ndim == 1:
            if x.size == 2 and y.size == 2:
                style = 'image'
            else:
                dx = np.diff(x)
                dy = np.diff(y)
                if np.ptp(dx) < 0.01 * abs(dx.mean()) and np.ptp(dy) < 0.01 * abs(dy.mean()):
                    style = 'image'
                else:
                    style = 'pcolorimage'
        elif x.ndim == 2 and y.ndim == 2:
            style = 'quadmesh'
        else:
            raise TypeError('arguments do not match valid signatures')
    else:
        raise _api.nargs_error('pcolorfast', '1 or 3', len(args))
    if style == 'quadmesh':
        coords = np.stack([x, y], axis=-1)
        if np.ndim(C) not in {2, 3}:
            raise ValueError('C must be 2D or 3D')
        collection = mcoll.QuadMesh(coords, array=C, alpha=alpha, cmap=cmap, norm=norm, antialiased=False, edgecolors='none')
        self.add_collection(collection, autolim=False)
        xl, xr, yb, yt = (x.min(), x.max(), y.min(), y.max())
        ret = collection
    else:
        extent = xl, xr, yb, yt = (x[0], x[-1], y[0], y[-1])
        if style == 'image':
            im = mimage.AxesImage(self, cmap=cmap, norm=norm, data=C, alpha=alpha, extent=extent, interpolation='nearest', origin='lower', **kwargs)
        elif style == 'pcolorimage':
            im = mimage.PcolorImage(self, x, y, C, cmap=cmap, norm=norm, alpha=alpha, extent=extent, **kwargs)
        self.add_image(im)
        ret = im
    if np.ndim(C) == 2:
        ret._scale_norm(norm, vmin, vmax)
    if ret.get_clip_path() is None:
        ret.set_clip_path(self.patch)
    ret.sticky_edges.x[:] = [xl, xr]
    ret.sticky_edges.y[:] = [yb, yt]
    self.update_datalim(np.array([[xl, yb], [xr, yt]]))
    self._request_autoscale_view(tight=True)
    return ret