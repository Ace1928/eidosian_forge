from __future__ import annotations
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs
from ..exceptions import PlotnineError, PlotnineWarning
def tdist_ci(x, dof, stderr, level):
    """
    Confidence Intervals using the t-distribution
    """
    import scipy.stats as stats
    q = (1 + level) / 2
    if dof is None:
        delta = stats.norm.ppf(q) * stderr
    else:
        delta = stats.t.ppf(q, dof) * stderr
    return (x - delta, x + delta)