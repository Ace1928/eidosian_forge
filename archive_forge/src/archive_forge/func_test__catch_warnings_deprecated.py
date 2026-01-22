import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
def test__catch_warnings_deprecated(self):
    with self.assertWarns(DeprecationWarning):
        with self._catch_warnings():
            pass