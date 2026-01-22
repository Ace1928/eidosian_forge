from __future__ import annotations
import typing
from copy import deepcopy
import numpy as np
from .._utils import jitter, resolution
from .position import position

    Jitter points to avoid overplotting

    Parameters
    ----------
    width :
        Proportion to jitter in horizontal direction.
        If `None`, `0.4` of the resolution of the data.
    height :
        Proportion to jitter in vertical direction.
        If `None`, `0.4` of the resolution of the data.
    random_state :
        Seed or Random number generator to use. If `None`, then
        numpy global generator [](`numpy.random`) is used.
    