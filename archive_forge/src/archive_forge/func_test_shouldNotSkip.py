from unittest import skipIf
from twisted.trial.unittest import TestCase
def test_shouldNotSkip(self) -> None:
    self.assertTrue(True, 'Test should run and not be skipped')