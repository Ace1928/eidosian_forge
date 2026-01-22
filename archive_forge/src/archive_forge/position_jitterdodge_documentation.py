from __future__ import annotations
import typing
from contextlib import suppress
from copy import copy
from .._utils import jitter, resolution
from ..exceptions import PlotnineError
from ..mapping.aes import SCALED_AESTHETICS
from .position import position
from .position_dodge import position_dodge

    Dodge and jitter to minimise overlap

    Useful when aligning points generated through
    [](`~plotnine.geoms.geom_point`) with dodged a
    [](`~plotnine.geoms.geom_boxplot`).

    Parameters
    ----------
    jitter_width :
        Proportion to jitter in horizontal direction.
        If `None`, `0.4` of the resolution of the data.
    jitter_height :
        Proportion to jitter in vertical direction.
    dodge_width :
        Amount to dodge in horizontal direction.
    random_state :
        Seed or Random number generator to use. If `None`, then
        numpy global generator [](`numpy.random`) is used.
    