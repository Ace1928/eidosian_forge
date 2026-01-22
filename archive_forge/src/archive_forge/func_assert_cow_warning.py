from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import (
import uuid
from pandas._config import using_copy_on_write
from pandas.compat import PYPY
from pandas.errors import ChainedAssignmentError
from pandas import set_option
from pandas.io.common import get_handle
def assert_cow_warning(warn=True, match=None, **kwargs):
    """
    Assert that a warning is raised in the CoW warning mode.

    Parameters
    ----------
    warn : bool, default True
        By default, check that a warning is raised. Can be turned off by passing False.
    match : str
        The warning message to match against, if different from the default.
    kwargs
        Passed through to assert_produces_warning
    """
    from pandas._testing import assert_produces_warning
    if not warn:
        from contextlib import nullcontext
        return nullcontext()
    if not match:
        match = 'Setting a value on a view'
    return assert_produces_warning(FutureWarning, match=match, **kwargs)