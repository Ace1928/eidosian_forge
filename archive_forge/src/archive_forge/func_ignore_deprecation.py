import contextlib
import os
import platform
import socket
import sys
import textwrap
import typing  # noqa: F401
import unittest
import warnings
from tornado.testing import bind_unused_port
@contextlib.contextmanager
def ignore_deprecation():
    """Context manager to ignore deprecation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        yield