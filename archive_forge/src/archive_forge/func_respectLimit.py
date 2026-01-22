import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def respectLimit():
    stats = team.statistics()
    if stats.busyWorkerCount + stats.idleWorkerCount >= currentLimit():
        return None
    return self._newWorker()