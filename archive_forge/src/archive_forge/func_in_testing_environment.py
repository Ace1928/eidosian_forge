import logging
import functools
import inspect
import itertools
import sys
import textwrap
import types
from pyomo.common.errors import DeveloperError
def in_testing_environment():
    """Return True if we are currently running in a "testing" environment

    This currently includes if nose, nose2, pytest, or Sphinx are
    running (imported).

    """
    return any((mod in sys.modules for mod in ('nose', 'nose2', 'pytest', 'sphinx')))