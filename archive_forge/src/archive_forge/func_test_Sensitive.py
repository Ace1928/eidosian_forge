from __future__ import annotations
import os
import sys
import types
from typing_extensions import NoReturn
from twisted.python import rebuild
from twisted.trial.unittest import TestCase
from . import crash_test_dummy
def test_Sensitive(self) -> None:
    """
        L{twisted.python.rebuild.Sensitive}
        """
    from twisted.python import rebuild
    from twisted.python.rebuild import Sensitive

    class TestSensitive(Sensitive):

        def test_method(self) -> None:
            """
                Dummy method
                """
    testSensitive = TestSensitive()
    testSensitive.rebuildUpToDate()
    self.assertFalse(testSensitive.needRebuildUpdate())
    newException = rebuild.latestClass(Exception)
    self.assertEqual(repr(Exception), repr(newException))
    self.assertEqual(newException, testSensitive.latestVersionOf(newException))
    self.assertEqual(TestSensitive.test_method, testSensitive.latestVersionOf(TestSensitive.test_method))
    self.assertEqual(testSensitive.test_method, testSensitive.latestVersionOf(testSensitive.test_method))
    self.assertEqual(TestSensitive, testSensitive.latestVersionOf(TestSensitive))

    def myFunction() -> None:
        """
            Dummy method
            """
    self.assertEqual(myFunction, testSensitive.latestVersionOf(myFunction))