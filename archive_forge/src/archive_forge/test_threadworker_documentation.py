import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker

        If L{LockWorker.do} is called concurrently with L{LockWorker.quit}, and
        C{quit} wins the race before C{do} gets the lock attribute, then
        L{AlreadyQuit} will be raised.
        