import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
def test_assert_deprecated_when_warning_already_issued(self):

    def old_and_dull_caller():
        old_and_dull()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always', DeprecationWarning)
        old_and_dull_caller()
        with self.assertDeprecated():
            old_and_dull_caller()