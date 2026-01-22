from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
def test_quitConcurrentWithWorkHappening(self):
    """
        If work happens after L{Team.quit} sets its C{Quit} flag, but before
        any other work takes place, the L{Team} should still exit gracefully.
        """
    self.team.do(list)
    originalSet = self.team._quit.set

    def performWorkConcurrently():
        originalSet()
        self.performAllOutstandingWork()
    self.team._quit.set = performWorkConcurrently
    self.team.quit()
    self.assertRaises(AlreadyQuit, self.team.quit)
    self.assertRaises(AlreadyQuit, self.team.do, list)