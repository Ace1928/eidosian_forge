import os
import re
import struct
import threading
import functools
import inspect
from tempfile import NamedTemporaryFile
import numpy as np
from numpy import testing
from numpy.testing import (
from .. import data, io
from ..data._fetchers import _fetch
from ..util import img_as_uint, img_as_float, img_as_int, img_as_ubyte
from ._warnings import expected_warnings
import pytest
def assert_stacklevel(warnings, *, offset=-1):
    """Assert correct stacklevel of captured warnings.

    When scikit-image raises warnings, the stacklevel should ideally be set
    so that the origin of the warnings will point to the public function
    that was called by the user and not necessarily the very place where the
    warnings were emitted (which may be inside of some internal function).
    This utility function helps with checking that
    the stacklevel was set correctly on warnings captured by `pytest.warns`.

    Parameters
    ----------
    warnings : collections.abc.Iterable[warning.WarningMessage]
        Warnings that were captured by `pytest.warns`.
    offset : int, optional
        Offset from the line this function is called to the line were the
        warning is supposed to originate from. For multiline calls, the
        first line is relevant. Defaults to -1 which corresponds to the line
        right above the one where this function is called.

    Raises
    ------
    AssertionError
        If a warning in `warnings` does not match the expected line number or
        file name.

    Examples
    --------
    >>> def test_something():
    ...     with pytest.warns(UserWarning, match="some message") as record:
    ...         something_raising_a_warning()
    ...     assert_stacklevel(record)
    ...
    >>> def test_another_thing():
    ...     with pytest.warns(UserWarning, match="some message") as record:
    ...         iam_raising_many_warnings(
    ...             "A long argument that forces the call to wrap."
    ...         )
    ...     assert_stacklevel(record, offset=-3)
    """
    frame = inspect.stack()[1].frame
    line_number = frame.f_lineno + offset
    filename = frame.f_code.co_filename
    expected = f'{filename}:{line_number}'
    for warning in warnings:
        actual = f'{warning.filename}:{warning.lineno}'
        assert actual == expected, f'{actual} != {expected}'