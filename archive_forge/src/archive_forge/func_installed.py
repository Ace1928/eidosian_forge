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
def installed(*specifiers):
    """Returns True if the current environment satisfies the specified
    package requirement.

    Args:
        specifiers: Version specifiers (e.g., `numpy>=1.20.0`).
    """
    import pkg_resources
    for spec in specifiers:
        try:
            pkg_resources.require(spec)
        except pkg_resources.ResolutionError:
            return False
    return True