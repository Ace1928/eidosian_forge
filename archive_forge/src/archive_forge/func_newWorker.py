import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def newWorker():
    self.workers.append(createMemoryWorker())
    return self.workers[-1][0]