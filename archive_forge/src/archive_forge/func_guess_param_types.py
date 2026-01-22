import collections
import copy
import datetime as dt
import glob
import inspect
import numbers
import os.path
import pathlib
import re
import sys
import typing
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from .parameterized import (
from ._utils import (
def guess_param_types(**kwargs):
    """
    Given a set of keyword literals, promote to the appropriate
    parameter type based on some simple heuristics.
    """
    params = {}
    for k, v in kwargs.items():
        kws = dict(default=v, constant=True)
        if isinstance(v, Parameter):
            params[k] = v
        elif isinstance(v, dt_types):
            params[k] = Date(**kws)
        elif isinstance(v, bool):
            params[k] = Boolean(**kws)
        elif isinstance(v, int):
            params[k] = Integer(**kws)
        elif isinstance(v, float):
            params[k] = Number(**kws)
        elif isinstance(v, str):
            params[k] = String(**kws)
        elif isinstance(v, dict):
            params[k] = Dict(**kws)
        elif isinstance(v, tuple):
            if all((_is_number(el) for el in v)):
                params[k] = NumericTuple(**kws)
            elif all((isinstance(el, dt_types) for el in v)) and len(v) == 2:
                params[k] = DateRange(**kws)
            else:
                params[k] = Tuple(**kws)
        elif isinstance(v, list):
            params[k] = List(**kws)
        else:
            if 'numpy' in sys.modules:
                from numpy import ndarray
                if isinstance(v, ndarray):
                    params[k] = Array(**kws)
                    continue
            if 'pandas' in sys.modules:
                from pandas import DataFrame as pdDFrame, Series as pdSeries
                if isinstance(v, pdDFrame):
                    params[k] = DataFrame(**kws)
                    continue
                elif isinstance(v, pdSeries):
                    params[k] = Series(**kws)
                    continue
            params[k] = Parameter(**kws)
    return params