from __future__ import annotations
from contextlib import contextmanager
import copy
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas import (
import pandas.core.common as com
from pandas.core.frame import (
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats.format import save_to_buffer
from pandas.io.formats.style_render import (
def css_calc(x, left: float, right: float, align: str, color: str | list | tuple):
    """
        Return the correct CSS for bar placement based on calculated values.

        Parameters
        ----------
        x : float
            Value which determines the bar placement.
        left : float
            Value marking the left side of calculation, usually minimum of data.
        right : float
            Value marking the right side of the calculation, usually maximum of data
            (left < right).
        align : {"left", "right", "zero", "mid"}
            How the bars will be positioned.
            "left", "right", "zero" can be used with any values for ``left``, ``right``.
            "mid" can only be used where ``left <= 0`` and ``right >= 0``.
            "zero" is used to specify a center when all values ``x``, ``left``,
            ``right`` are translated, e.g. by say a mean or median.

        Returns
        -------
        str : Resultant CSS with linear gradient.

        Notes
        -----
        Uses ``colors``, ``width`` and ``height`` from outer scope.
        """
    if pd.isna(x):
        return base_css
    if isinstance(color, (list, tuple)):
        color = color[0] if x < 0 else color[1]
    assert isinstance(color, str)
    x = left if x < left else x
    x = right if x > right else x
    start: float = 0
    end: float = 1
    if align == 'left':
        end = (x - left) / (right - left)
    elif align == 'right':
        start = (x - left) / (right - left)
    else:
        z_frac: float = 0.5
        if align == 'zero':
            limit: float = max(abs(left), abs(right))
            left, right = (-limit, limit)
        elif align == 'mid':
            mid: float = (left + right) / 2
            z_frac = -mid / (right - left) + 0.5 if mid < 0 else -left / (right - left)
        if x < 0:
            start, end = ((x - left) / (right - left), z_frac)
        else:
            start, end = (z_frac, (x - left) / (right - left))
    ret = css_bar(start * width, end * width, color)
    if height < 1 and 'background: linear-gradient(' in ret:
        return ret + f' no-repeat center; background-size: 100% {height * 100:.1f}%;'
    else:
        return ret