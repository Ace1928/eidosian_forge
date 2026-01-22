import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
def to_scalar_or_list(v):
    np = get_module('numpy', should_load=False)
    pd = get_module('pandas', should_load=False)
    if np and np.isscalar(v) and hasattr(v, 'item'):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [to_scalar_or_list(e) for e in v]
    elif np and isinstance(v, np.ndarray):
        if v.ndim == 0:
            return v.item()
        return [to_scalar_or_list(e) for e in v]
    elif pd and isinstance(v, (pd.Series, pd.Index)):
        return [to_scalar_or_list(e) for e in v]
    elif is_numpy_convertable(v):
        return to_scalar_or_list(np.array(v))
    else:
        return v