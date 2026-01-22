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
def modulus_trans(p, offset=1, **kwargs):
    """
    Modulus Transformation

    The modulus transformation generalises Box-Cox to work with
    both positive and negative values.

    When :math:`y \\neq 0`

    .. math::

        y^{(\\lambda)} = sign(y) * \\frac{(|y| + 1)^\\lambda - 1}{\\lambda}

    and when :math:`y = 0`

    .. math::

        y^{(\\lambda)} =  sign(y) * \\ln{(|y| + 1)}

    Parameters
    ----------
    p : float
        Transformation exponent :math:`\\lambda`.
    offset : int
        Constant offset. 0 for Box-Cox type 1, otherwise any
        non-negative constant (Box-Cox type 2).
        The default is 1. :func:`~mizani.transforms.boxcox_trans`
        sets the default to 0.
    kwargs : dict
        Keyword arguments passed onto :func:`trans_new`.
        Should not include the `transform` or `inverse`.

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
    :func:`~mizani.transforms.boxcox_trans`
    """
    if np.abs(p) < 1e-07:

        def transform(x: FloatArrayLike) -> NDArrayFloat:
            x = np.asarray(x)
            return np.sign(x) * np.log(np.abs(x) + offset)

        def inverse(x: FloatArrayLike) -> NDArrayFloat:
            x = np.asarray(x)
            return np.sign(x) * (np.exp(np.abs(x)) - offset)
    else:

        def transform(x: FloatArrayLike) -> NDArrayFloat:
            x = np.asarray(x)
            return np.sign(x) * ((np.abs(x) + offset) ** p - 1) / p

        def inverse(x: FloatArrayLike) -> NDArrayFloat:
            x = np.asarray(x)
            return np.sign(x) * ((np.abs(x) * p + 1) ** (1 / p) - offset)
    kwargs['p'] = p
    kwargs['offset'] = offset
    kwargs['name'] = kwargs.get('name', 'mt_pow_{}'.format(p))
    kwargs['transform'] = transform
    kwargs['inverse'] = inverse
    return trans_new(**kwargs)