from __future__ import annotations
import ast
import base64
import datetime as dt
import json
import logging
import numbers
import os
import pathlib
import re
import sys
import urllib.parse as urlparse
from collections import OrderedDict, defaultdict
from collections.abc import MutableMapping, MutableSequence
from datetime import datetime
from functools import partial
from html import escape  # noqa
from importlib import import_module
from typing import Any, AnyStr
import bokeh
import numpy as np
import param
from bokeh.core.has_props import _default_resolver
from bokeh.model import Model
from packaging.version import Version
from .checks import (  # noqa
from .parameters import (  # noqa
def param_reprs(parameterized, skip=None):
    """
    Returns a list of reprs for parameters on the parameterized object.
    Skips default and empty values.
    """
    cls = type(parameterized).__name__
    param_reprs = []
    for p, v in sorted(parameterized.param.values().items()):
        default = parameterized.param[p].default
        equal = v is default
        if not equal:
            if isinstance(v, np.ndarray):
                if isinstance(default, np.ndarray):
                    equal = np.array_equal(v, default, equal_nan=True)
                else:
                    equal = False
            else:
                try:
                    equal = bool(v == default)
                except Exception:
                    equal = False
        if equal:
            continue
        elif v is None:
            continue
        elif isinstance(v, str) and v == '':
            continue
        elif isinstance(v, list) and v == []:
            continue
        elif isinstance(v, dict) and v == {}:
            continue
        elif skip and p in skip or (p == 'name' and v.startswith(cls)):
            continue
        else:
            v = abbreviated_repr(v)
        param_reprs.append(f'{p}={v}')
    return param_reprs