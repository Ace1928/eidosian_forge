from twisted.trial.unittest import SynchronousTestCase
from .._convenience import Quit
from .._ithreads import AlreadyQuit
def test_checkDoesNothing(self) -> None:
    """
        L{Quit.check} initially does nothing and returns L{None}.
        """
    quit = Quit()
    self.assertIs(quit.check(), None)