import functools
import logging
import os
import pathlib
import sys
import sysconfig
from typing import Any, Dict, Generator, Optional, Tuple
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.virtualenv import running_under_virtualenv
from . import _sysconfig
from .base import (
def get_platlib() -> str:
    """Return the default platform-shared lib location."""
    new = _sysconfig.get_platlib()
    if _USE_SYSCONFIG:
        return new
    from . import _distutils
    old = _distutils.get_platlib()
    if _looks_like_deb_system_dist_packages(old):
        return old
    if _warn_if_mismatch(pathlib.Path(old), pathlib.Path(new), key='platlib'):
        _log_context()
    return old