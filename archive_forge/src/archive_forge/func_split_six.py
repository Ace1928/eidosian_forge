import base64
import json
import math
import os
import re
import struct
import typing
import zlib
from typing import Any, Callable, Union
from jinja2 import Environment, PackageLoader
def split_six(series=None):
    """
    Given a Pandas Series, get a domain of values from zero to the 90% quantile
    rounded to the nearest order-of-magnitude integer. For example, 2100 is
    rounded to 2000, 2790 to 3000.

    Parameters
    ----------
    series: Pandas series, default None

    Returns
    -------
    list

    """
    if pd is None:
        raise ImportError('The Pandas package is required for this functionality')
    if np is None:
        raise ImportError('The NumPy package is required for this functionality')

    def base(x):
        if x > 0:
            base = pow(10, math.floor(math.log10(x)))
            return round(x / base) * base
        else:
            return 0
    quants = [0, 50, 75, 85, 90]
    arr = series.values
    return [base(np.percentile(arr, x)) for x in quants]