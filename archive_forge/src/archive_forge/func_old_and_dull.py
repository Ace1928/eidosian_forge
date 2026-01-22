import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
@deprecated("This function is outdated. Use 'shiny' instead!")
def old_and_dull():
    """ A deprecated function, for use in assertDeprecated tests.

    """
    pass