import numpy as np
import pandas as pd
from ..doctools import document
from ..mapping.evaluation import after_stat
from .stat import stat

    Emperical Cumulative Density Function

    {usage}

    Parameters
    ----------
    {common_parameters}
    n  : int, default=None
        This is the number of points to interpolate with.
        If `None`{.py}, do not interpolate.
    pad : bool, default=True
        If True, pad the domain with `-inf` and `+inf` so that
        ECDF does not have discontinuities at the extremes.

    See Also
    --------
    plotnine.geom_step
    