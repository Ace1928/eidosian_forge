import unittest as pyunit
from twisted.trial import itrial, reporter, runner, unittest
from twisted.trial.test import mockdoctest
def test_expectedResults(self, count: int=1) -> None:
    """
        Trial can correctly run doctests with its xUnit test APIs.
        """
    suite = runner.TestLoader().loadDoctests(mockdoctest)
    self._testRun(suite)