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
def is_homogeneous_array(v):
    """
    Return whether a value is considered to be a homogeneous array
    """
    np = get_module('numpy', should_load=False)
    pd = get_module('pandas', should_load=False)
    if np and isinstance(v, np.ndarray) or (pd and isinstance(v, (pd.Series, pd.Index))):
        return True
    if is_numpy_convertable(v):
        np = get_module('numpy', should_load=True)
        if np:
            v_numpy = np.array(v)
            if v_numpy.shape == ():
                return False
            else:
                return True
    return False