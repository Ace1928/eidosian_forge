from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.evaluation import after_stat
from .stat import stat

    Port of R stats::bw.nrd0

    This is equivalent to statsmodels silverman when x has more than
    1 unique value. It can never give a zero bandwidth.

    Parameters
    ----------
    x : array_like
        Values whose density is to be estimated

    Returns
    -------
    out : float
        Bandwidth of x
    