from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from ..exceptions import PlotnineError
from ..scales.scale_discrete import scale_discrete
def breaks_from_binwidth(x_range: TupleFloat2, binwidth: float, center: Optional[float]=None, boundary: Optional[float]=None):
    """
    Calculate breaks given binwidth

    Parameters
    ----------
    x_range :
        Range over with to calculate the breaks. Must be
        of size 2.
    binwidth :
        Separation between the breaks
    center :
        The center of one of the bins
    boundary :
        A boundary between two bins

    Returns
    -------
    out : array_like
        Sequence of break points.
    """
    if binwidth <= 0:
        raise PlotnineError("The 'binwidth' must be positive.")
    if boundary is not None and center is not None:
        raise PlotnineError("Only one of 'boundary' and 'center' may be specified.")
    elif boundary is None:
        boundary = binwidth / 2
        if center is not None:
            boundary = center - boundary
    epsilon = np.finfo(float).eps
    shift = np.floor((x_range[0] - boundary) / binwidth)
    origin = boundary + shift * binwidth
    max_x = x_range[1] + binwidth * (1 - epsilon)
    breaks = np.arange(origin, max_x, binwidth)
    return breaks