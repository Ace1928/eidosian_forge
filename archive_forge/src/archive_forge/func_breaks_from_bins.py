from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from ..exceptions import PlotnineError
from ..scales.scale_discrete import scale_discrete
def breaks_from_bins(x_range: TupleFloat2, bins: int=30, center: Optional[float]=None, boundary: Optional[float]=None):
    """
    Calculate breaks given binwidth

    Parameters
    ----------
    x_range :
        Range over with to calculate the breaks. Must be
        of size 2.
    bins :
        Number of bins
    center :
        The center of one of the bins
    boundary :
        A boundary between two bins

    Returns
    -------
    out : array_like
        Sequence of break points.
    """
    if bins < 1:
        raise PlotnineError('Need at least one bin.')
    elif bins == 1:
        binwidth = x_range[1] - x_range[0]
        boundary = x_range[1]
    else:
        binwidth = (x_range[1] - x_range[0]) / (bins - 1)
    return breaks_from_binwidth(x_range, binwidth, center, boundary)