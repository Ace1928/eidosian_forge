import sys
import weakref
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
from .core import util
from .core.ndmapping import UniformNdMapping
def streams_list_from_dict(streams):
    """Converts a streams dictionary into a streams list"""
    params = {}
    for k, v in streams.items():
        if 'panel' in sys.modules:
            if util.param_version > util.Version('2.0.0rc1'):
                v = param.parameterized.transform_reference(v)
            else:
                from panel.depends import param_value_if_widget
                v = param_value_if_widget(v)
        if isinstance(v, param.Parameter) and v.owner is not None:
            params[k] = v
        else:
            raise TypeError(f'Cannot handle value {v!r} in streams dictionary')
    return Params.from_params(params)