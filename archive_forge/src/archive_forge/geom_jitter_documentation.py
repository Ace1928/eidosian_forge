from __future__ import annotations
import typing
from ..doctools import document
from ..exceptions import PlotnineError
from ..positions import position_jitter
from .geom_point import geom_point

    Scatter plot with points jittered to reduce overplotting

    {usage}

    Parameters
    ----------
    {common_parameters}
    width : float, default=None
        Proportion to jitter in horizontal direction.
        The default value is that from
        [](`~plotnine.positions.position_jitter`)
    height : float, default=None
        Proportion to jitter in vertical direction.
        The default value is that from
        [](`~plotnine.positions.position_jitter`).
    random_state : int | ~numpy.random.RandomState, default=None
        Seed or Random number generator to use. If `None`, then
        numpy global generator [](`numpy.random`) is used.

    See Also
    --------
    plotnine.position_jitter
    plotnine.geom_point
    