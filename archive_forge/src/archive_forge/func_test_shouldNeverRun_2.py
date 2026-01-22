from unittest import skipIf
from twisted.trial.unittest import TestCase
def test_shouldNeverRun_2(self) -> None:
    raise Exception('Test should skip and never reach here')