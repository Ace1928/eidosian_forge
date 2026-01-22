import contextlib
import inspect
from typing import Callable
import unittest
from unittest import mock
import warnings
import numpy
import cupy
from cupy._core import internal
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
def numpy_satisfies(version_range):
    """Returns True if numpy version satisfies the specified criteria.

    Args:
        version_range: A version specifier (e.g., `>=1.13.0`).
    """
    return installed('numpy{}'.format(version_range))