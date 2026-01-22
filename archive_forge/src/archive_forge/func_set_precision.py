import warnings
from enum import IntEnum
import numpy as np
from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import multithreading_enabled, requires_geos
@requires_geos('3.6.0')
@multithreading_enabled
def set_precision(geometry, grid_size, mode='valid_output', **kwargs):
    """Returns geometry with the precision set to a precision grid size.

    By default, geometries use double precision coordinates (grid_size = 0).

    Coordinates will be rounded if a precision grid is less precise than the
    input geometry. Duplicated vertices will be dropped from lines and
    polygons for grid sizes greater than 0. Line and polygon geometries may
    collapse to empty geometries if all vertices are closer together than
    grid_size. Z values, if present, will not be modified.

    Note: subsequent operations will always be performed in the precision of
    the geometry with higher precision (smaller "grid_size"). That same
    precision will be attached to the operation outputs.

    Also note: input geometries should be geometrically valid; unexpected
    results may occur if input geometries are not.

    Returns None if geometry is None.

    Parameters
    ----------
    geometry : Geometry or array_like
    grid_size : float
        Precision grid size. If 0, will use double precision (will not modify
        geometry if precision grid size was not previously set). If this
        value is more precise than input geometry, the input geometry will
        not be modified.
    mode :  {'valid_output', 'pointwise', 'keep_collapsed'}, default 'valid_output'
        This parameter determines how to handle invalid output geometries. There are three modes:

        1. `'valid_output'` (default):  The output is always valid. Collapsed geometry elements
           (including both polygons and lines) are removed. Duplicate vertices are removed.
        2. `'pointwise'`: Precision reduction is performed pointwise. Output geometry
           may be invalid due to collapse or self-intersection. Duplicate vertices are not
           removed. In GEOS this option is called NO_TOPO.

           .. note::

             'pointwise' mode requires at least GEOS 3.10. It is accepted in earlier versions,
             but the results may be unexpected.
        3. `'keep_collapsed'`: Like the default mode, except that collapsed linear geometry
           elements are preserved. Collapsed polygonal input elements are removed. Duplicate
           vertices are removed.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    get_precision

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> set_precision(Point(0.9, 0.9), 1.0)
    <POINT (1 1)>
    >>> set_precision(Point(0.9, 0.9, 0.9), 1.0)
    <POINT Z (1 1 0.9)>
    >>> set_precision(LineString([(0, 0), (0, 0.1), (0, 1), (1, 1)]), 1.0)
    <LINESTRING (0 0, 0 1, 1 1)>
    >>> set_precision(LineString([(0, 0), (0, 0.1), (0.1, 0.1)]), 1.0, mode="valid_output")
    <LINESTRING Z EMPTY>
    >>> set_precision(LineString([(0, 0), (0, 0.1), (0.1, 0.1)]), 1.0, mode="pointwise")
    <LINESTRING (0 0, 0 0, 0 0)>
    >>> set_precision(LineString([(0, 0), (0, 0.1), (0.1, 0.1)]), 1.0, mode="keep_collapsed")
    <LINESTRING (0 0, 0 0)>
    >>> set_precision(None, 1.0) is None
    True
    """
    if isinstance(mode, str):
        mode = SetPrecisionMode.get_value(mode)
    elif not np.isscalar(mode):
        raise TypeError('mode only accepts scalar values')
    if mode == SetPrecisionMode.pointwise and geos_version < (3, 10, 0):
        warnings.warn("'pointwise' is only supported for GEOS 3.10", UserWarning, stacklevel=2)
    return lib.set_precision(geometry, grid_size, np.intc(mode), **kwargs)