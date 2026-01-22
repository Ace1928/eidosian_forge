import unittest as pyunit
from twisted.trial import itrial, reporter, runner, unittest
from twisted.trial.test import mockdoctest
def test_repeatable(self) -> None:
    """
        Doctests should be runnable repeatably.
        """
    suite = runner.TestLoader().loadDoctests(mockdoctest)
    self._testRun(suite)
    self._testRun(suite)