import inspect
import types
import traceback
import sys
import operator as op
from collections import namedtuple
import warnings
import re
import builtins
import typing
from pathlib import Path
from typing import Optional, Tuple
from jedi.inference.compiled.getattr_static import getattr_static
def safe_getattr(obj, name, default=_sentinel):
    try:
        attr, is_get_descriptor = getattr_static(obj, name)
    except AttributeError:
        if default is _sentinel:
            raise
        return default
    else:
        if isinstance(attr, ALLOWED_DESCRIPTOR_ACCESS):
            return getattr(obj, name)
    return attr