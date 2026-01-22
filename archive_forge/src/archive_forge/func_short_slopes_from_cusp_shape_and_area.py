from ..sage_helper import _within_sage
import math
def short_slopes_from_cusp_shape_and_area(cusp_shape, cusp_area, length=6):
    """
    cusp_shape is longitude over meridian (conjugate).
    l/m

    sage: from sage.all import RIF, CIF
    sage: short_slopes_from_cusp_shape_and_area(CIF(RIF(1.0),RIF(1.3333,1.3334)), RIF(12.0))
    [(1, 0), (-2, 1), (-1, 1), (0, 1)]

    >>> short_slopes_from_cusp_shape_and_area(1.0+1.3333j, 12.0)
    [(1, 0), (-2, 1), (-1, 1), (0, 1)]

    """
    return short_slopes_from_translations(translations_from_cusp_shape_and_area(cusp_shape, cusp_area), length)