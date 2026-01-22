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
def pcolor(self, *args, shading=None, alpha=None, norm=None, cmap=None, vmin=None, vmax=None, **kwargs):
    """
        Create a pseudocolor plot with a non-regular rectangular grid.

        Call signature::

            pcolor([X, Y,] C, **kwargs)

        *X* and *Y* can be used to specify the corners of the quadrilaterals.

        .. hint::

            ``pcolor()`` can be very slow for large arrays. In most
            cases you should use the similar but much faster
            `~.Axes.pcolormesh` instead. See
            :ref:`Differences between pcolor() and pcolormesh()
            <differences-pcolor-pcolormesh>` for a discussion of the
            differences.

        Parameters
        ----------
        C : 2D array-like
            The color-mapped values.  Color-mapping is controlled by *cmap*,
            *norm*, *vmin*, and *vmax*.

        X, Y : array-like, optional
            The coordinates of the corners of quadrilaterals of a pcolormesh::

                (X[i+1, j], Y[i+1, j])       (X[i+1, j+1], Y[i+1, j+1])
                                      ●╶───╴●
                                      │     │
                                      ●╶───╴●
                    (X[i, j], Y[i, j])       (X[i, j+1], Y[i, j+1])

            Note that the column index corresponds to the x-coordinate, and
            the row index corresponds to y. For details, see the
            :ref:`Notes <axes-pcolormesh-grid-orientation>` section below.

            If ``shading='flat'`` the dimensions of *X* and *Y* should be one
            greater than those of *C*, and the quadrilateral is colored due
            to the value at ``C[i, j]``.  If *X*, *Y* and *C* have equal
            dimensions, a warning will be raised and the last row and column
            of *C* will be ignored.

            If ``shading='nearest'``, the dimensions of *X* and *Y* should be
            the same as those of *C* (if not, a ValueError will be raised). The
            color ``C[i, j]`` will be centered on ``(X[i, j], Y[i, j])``.

            If *X* and/or *Y* are 1-D arrays or column vectors they will be
            expanded as needed into the appropriate 2D arrays, making a
            rectangular grid.

        shading : {'flat', 'nearest', 'auto'}, default: :rc:`pcolor.shading`
            The fill style for the quadrilateral. Possible values:

            - 'flat': A solid color is used for each quad. The color of the
              quad (i, j), (i+1, j), (i, j+1), (i+1, j+1) is given by
              ``C[i, j]``. The dimensions of *X* and *Y* should be
              one greater than those of *C*; if they are the same as *C*,
              then a deprecation warning is raised, and the last row
              and column of *C* are dropped.
            - 'nearest': Each grid point will have a color centered on it,
              extending halfway between the adjacent grid centers.  The
              dimensions of *X* and *Y* must be the same as *C*.
            - 'auto': Choose 'flat' if dimensions of *X* and *Y* are one
              larger than *C*.  Choose 'nearest' if dimensions are the same.

            See :doc:`/gallery/images_contours_and_fields/pcolormesh_grids`
            for more description.

        %(cmap_doc)s

        %(norm_doc)s

        %(vmin_vmax_doc)s

        edgecolors : {'none', None, 'face', color, color sequence}, optional
            The color of the edges. Defaults to 'none'. Possible values:

            - 'none' or '': No edge.
            - *None*: :rc:`patch.edgecolor` will be used. Note that currently
              :rc:`patch.force_edgecolor` has to be True for this to work.
            - 'face': Use the adjacent face color.
            - A color or sequence of colors will set the edge color.

            The singular form *edgecolor* works as an alias.

        alpha : float, default: None
            The alpha blending value of the face color, between 0 (transparent)
            and 1 (opaque). Note: The edgecolor is currently not affected by
            this.

        snap : bool, default: False
            Whether to snap the mesh to pixel boundaries.

        Returns
        -------
        `matplotlib.collections.PolyQuadMesh`

        Other Parameters
        ----------------
        antialiaseds : bool, default: False
            The default *antialiaseds* is False if the default
            *edgecolors*\\ ="none" is used.  This eliminates artificial lines
            at patch boundaries, and works regardless of the value of alpha.
            If *edgecolors* is not "none", then the default *antialiaseds*
            is taken from :rc:`patch.antialiased`.
            Stroking the edges may be preferred if *alpha* is 1, but will
            cause artifacts otherwise.

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Additionally, the following arguments are allowed. They are passed
            along to the `~matplotlib.collections.PolyQuadMesh` constructor:

        %(PolyCollection:kwdoc)s

        See Also
        --------
        pcolormesh : for an explanation of the differences between
            pcolor and pcolormesh.
        imshow : If *X* and *Y* are each equidistant, `~.Axes.imshow` can be a
            faster alternative.

        Notes
        -----
        **Masked arrays**

        *X*, *Y* and *C* may be masked arrays. If either ``C[i, j]``, or one
        of the vertices surrounding ``C[i, j]`` (*X* or *Y* at
        ``[i, j], [i+1, j], [i, j+1], [i+1, j+1]``) is masked, nothing is
        plotted.

        .. _axes-pcolor-grid-orientation:

        **Grid orientation**

        The grid orientation follows the standard matrix convention: An array
        *C* with shape (nrows, ncolumns) is plotted with the column number as
        *X* and the row number as *Y*.
        """
    if shading is None:
        shading = mpl.rcParams['pcolor.shading']
    shading = shading.lower()
    X, Y, C, shading = self._pcolorargs('pcolor', *args, shading=shading, kwargs=kwargs)
    linewidths = (0.25,)
    if 'linewidth' in kwargs:
        kwargs['linewidths'] = kwargs.pop('linewidth')
    kwargs.setdefault('linewidths', linewidths)
    if 'edgecolor' in kwargs:
        kwargs['edgecolors'] = kwargs.pop('edgecolor')
    ec = kwargs.setdefault('edgecolors', 'none')
    if 'antialiaseds' in kwargs:
        kwargs['antialiased'] = kwargs.pop('antialiaseds')
    if 'antialiased' not in kwargs and cbook._str_lower_equal(ec, 'none'):
        kwargs['antialiased'] = False
    kwargs.setdefault('snap', False)
    if np.ma.isMaskedArray(X) or np.ma.isMaskedArray(Y):
        stack = np.ma.stack
        X = np.ma.asarray(X)
        Y = np.ma.asarray(Y)
        x = X.compressed()
        y = Y.compressed()
    else:
        stack = np.stack
        x = X
        y = Y
    coords = stack([X, Y], axis=-1)
    collection = mcoll.PolyQuadMesh(coords, array=C, cmap=cmap, norm=norm, alpha=alpha, **kwargs)
    collection._scale_norm(norm, vmin, vmax)
    t = collection._transform
    if not isinstance(t, mtransforms.Transform) and hasattr(t, '_as_mpl_transform'):
        t = t._as_mpl_transform(self.axes)
    if t and any(t.contains_branch_seperately(self.transData)):
        trans_to_data = t - self.transData
        pts = np.vstack([x, y]).T.astype(float)
        transformed_pts = trans_to_data.transform(pts)
        x = transformed_pts[..., 0]
        y = transformed_pts[..., 1]
    self.add_collection(collection, autolim=False)
    minx = np.min(x)
    maxx = np.max(x)
    miny = np.min(y)
    maxy = np.max(y)
    collection.sticky_edges.x[:] = [minx, maxx]
    collection.sticky_edges.y[:] = [miny, maxy]
    corners = ((minx, miny), (maxx, maxy))
    self.update_datalim(corners)
    self._request_autoscale_view()
    return collection