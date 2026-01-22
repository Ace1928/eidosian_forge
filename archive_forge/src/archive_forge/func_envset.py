import os
import sys
import typing
from contextlib import contextmanager
from collections.abc import Iterable
from IPython import get_ipython
from traitlets import (
from json import loads as jsonloads, dumps as jsondumps
from .. import comm
from base64 import standard_b64encode
from .utils import deprecation, _get_frame
from .._version import __protocol_version__, __control_protocol_version__, __jupyter_widgets_base_version__
import inspect
def envset(name, default):
    """Return True if the given environment variable is turned on, otherwise False
    If the environment variable is set, True will be returned if it is assigned to a value
    other than 'no', 'n', 'false', 'off', '0', or '0.0' (case insensitive).
    If the environment variable is not set, the default value is returned.
    """
    if name in os.environ:
        return os.environ[name].lower() not in ['no', 'n', 'false', 'off', '0', '0.0']
    else:
        return bool(default)