from io import StringIO
from twisted.python.failure import Failure
from twisted.trial._dist.distreporter import DistReporter
from twisted.trial.reporter import TreeReporter
from twisted.trial.unittest import TestCase
def test_startErrorStop(self) -> None:
    """
        Error output only gets sent to the stream after the test has stopped.
        """
    self.distReporter.startTest(self.test)
    self.assertEqual(self.stream.getvalue(), '')
    self.distReporter.addError(self.test, Failure(Exception('error')))
    self.assertEqual(self.stream.getvalue(), '')
    self.distReporter.stopTest(self.test)
    self.assertNotEqual(self.stream.getvalue(), '')