from __future__ import annotations
import typing
from copy import copy
import pandas as pd
from matplotlib.animation import ArtistAnimation
from .exceptions import PlotnineError
def set_scale_limits(scales: list[scale]):
    """
            Set limits of all the scales in the animation

            Should be called before `check_scale_limits`.

            Parameters
            ----------
            scales : list[scales]
                List of scales the have been used in building a
                ggplot object.
            """
    for sc in scales:
        ae = sc.aesthetics[0]
        scale_limits[ae] = sc.limits