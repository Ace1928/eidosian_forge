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
def boxcox_trans(p, offset=0, **kwargs):
    """
    Boxcox Transformation

    The Box-Cox transformation is a flexible transformation, often
    used to transform data towards normality.

    The Box-Cox power transformation (type 1) requires strictly positive
    values and takes the following form for :math:`y \\gt 0`:

    .. math::

        y^{(\\lambda)} = \\frac{y^\\lambda - 1}{\\lambda}

    When :math:`y = 0`, the natural log transform is used.

    Parameters
    ----------
    p : float
        Transformation exponent :math:`\\lambda`.
    offset : int
        Constant offset. 0 for Box-Cox type 1, otherwise any
        non-negative constant (Box-Cox type 2).
        The default is 0. :func:`~mizani.transforms.modulus_trans`
        sets the default to 1.
    kwargs : dict
        Keyword arguments passed onto :func:`trans_new`. Should not
        include the `transform` or `inverse`.

    References
    ----------
    - Box, G. E., & Cox, D. R. (1964). An analysis of transformations.
      Journal of the Royal Statistical Society. Series B (Methodological),
      211-252. `<https://www.jstor.org/stable/2984418>`_
    - John, J. A., & Draper, N. R. (1980). An alternative family of
      transformations. Applied Statistics, 190-197.
      `<http://www.jstor.org/stable/2986305>`_

    See Also
    --------
    :func:`~mizani.transforms.modulus_trans`

    """

    def transform(x: FloatArrayLike) -> NDArrayFloat:
        x = np.asarray(x)
        if np.any(x + offset < 0):
            raise ValueError('boxcox_trans must be given only positive values. Consider using modulus_trans instead?')
        if np.abs(p) < 1e-07:
            return np.log(x + offset)
        else:
            return ((x + offset) ** p - 1) / p

    def inverse(x: FloatArrayLike) -> NDArrayFloat:
        x = np.asarray(x)
        if np.abs(p) < 1e-07:
            return np.exp(x) - offset
        else:
            return (x * p + 1) ** (1 / p) - offset
    kwargs['p'] = p
    kwargs['offset'] = offset
    kwargs['name'] = kwargs.get('name', 'pow_{}'.format(p))
    kwargs['transform'] = transform
    kwargs['inverse'] = inverse
    return trans_new(**kwargs)