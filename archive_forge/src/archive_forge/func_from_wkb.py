import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely._ragged_array import from_ragged_array, to_ragged_array
from shapely.decorators import requires_geos
from shapely.errors import UnsupportedGEOSVersionError
def from_wkb(geometry, on_invalid='raise', **kwargs):
    """
    Creates geometries from the Well-Known Binary (WKB) representation.

    The Well-Known Binary format is defined in the `OGC Simple Features
    Specification for SQL <https://www.opengeospatial.org/standards/sfs>`__.


    Parameters
    ----------
    geometry : str or array_like
        The WKB byte object(s) to convert.
    on_invalid : {"raise", "warn", "ignore"}, default "raise"
        - raise: an exception will be raised if a WKB input geometry is invalid.
        - warn: a warning will be raised and invalid WKB geometries will be
          returned as ``None``.
        - ignore: invalid WKB geometries will be returned as ``None`` without a warning.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from_wkb(b'\\x01\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\xf0?\\x00\\x00\\x00\\x00\\x00\\x00\\xf0?')
    <POINT (1 1)>
    """
    if not np.isscalar(on_invalid):
        raise TypeError('on_invalid only accepts scalar values')
    invalid_handler = np.uint8(DecodingErrorOptions.get_value(on_invalid))
    geometry = np.asarray(geometry, dtype=object)
    return lib.from_wkb(geometry, invalid_handler, **kwargs)