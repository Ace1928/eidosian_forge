from __future__ import annotations
import sys
import typing
from abc import ABC, abstractmethod
from datetime import MAXYEAR, MINYEAR, datetime, timedelta
from types import MethodType
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from ._core.dates import datetime_to_num, num_to_datetime
from .breaks import (
from .labels import (
from .utils import identity
def probability_trans(distribution: str, *args, **kwargs) -> trans:
    """
    Probability Transformation

    Parameters
    ----------
    distribution : str
        Name of the distribution. Valid distributions are
        listed at :mod:`scipy.stats`. Any of the continuous
        or discrete distributions.
    args : tuple
        Arguments passed to the distribution functions.
    kwargs : dict
        Keyword arguments passed to the distribution functions.

    Notes
    -----
    Make sure that the distribution is a good enough
    approximation for the data. When this is not the case,
    computations may run into errors. Absence of any errors
    does not imply that the distribution fits the data.
    """
    import scipy.stats as stats
    cdists = {k for k in dir(stats) if hasattr(getattr(stats, k), 'cdf')}
    if distribution not in cdists:
        raise ValueError(f"Unknown distribution '{distribution}'")
    try:
        doc = kwargs.pop('_doc')
    except KeyError:
        doc = ''
    try:
        name = kwargs.pop('_name')
    except KeyError:
        name = 'prob_{}'.format(distribution)

    def transform(x: FloatArrayLike) -> NDArrayFloat:
        return getattr(stats, distribution).cdf(x, *args, **kwargs)

    def inverse(x: FloatArrayLike) -> NDArrayFloat:
        return getattr(stats, distribution).ppf(x, *args, **kwargs)
    return trans_new(name, transform, inverse, domain=(0, 1), doc=doc)