import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker
def test_quitWhileGettingLock(self):
    """
        If L{LockWorker.do} is called concurrently with L{LockWorker.quit}, and
        C{quit} wins the race before C{do} gets the lock attribute, then
        L{AlreadyQuit} will be raised.
        """

    class RacyLockWorker(LockWorker):

        @property
        def _lock(self):
            self.quit()
            return self.__dict__['_lock']

        @_lock.setter
        def _lock(self, value):
            self.__dict__['_lock'] = value
    worker = RacyLockWorker(FakeLock(), local())
    self.assertRaises(AlreadyQuit, worker.do, list)