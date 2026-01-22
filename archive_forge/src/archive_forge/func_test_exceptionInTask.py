from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
def test_exceptionInTask(self):
    """
        When an exception is raised in a task passed to L{Team.do}, the
        C{logException} given to the L{Team} at construction is invoked in the
        exception context.
        """
    self.team.do(lambda: 1 / 0)
    self.performAllOutstandingWork()
    self.assertEqual(len(self.failures), 1)
    self.assertEqual(self.failures[0].type, ZeroDivisionError)