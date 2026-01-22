from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
import pandas as pd
from mizani.bounds import censor, expand_range_distinct, rescale, zero_range
from .._utils import match
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..iapi import range_view, scale_view
from ._expand import expand_range
from .range import RangeContinuous
from .scale import scale
def get_bounded_breaks(self, limits: Optional[ScaleContinuousLimits]=None) -> ScaleContinuousBreaks:
    """
        Return Breaks that are within limits
        """
    if limits is None:
        limits = self.limits
    breaks = self.get_breaks(limits)
    strict_breaks = [b for b in breaks if limits[0] <= b <= limits[1]]
    return strict_breaks