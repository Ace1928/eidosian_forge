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
def guess_bounds(params, **overrides):
    """
    Given a dictionary of Parameter instances, return a corresponding
    set of copies with the bounds appropriately set.


    If given a set of override keywords, use those numeric tuple bounds.
    """
    guessed = {}
    for name, p in params.items():
        new_param = copy.copy(p)
        if isinstance(p, (Integer, Number)):
            if name in overrides:
                minv, maxv = overrides[name]
            else:
                minv, maxv, _ = _get_min_max_value(None, None, value=p.default)
            new_param.bounds = (minv, maxv)
        guessed[name] = new_param
    return guessed