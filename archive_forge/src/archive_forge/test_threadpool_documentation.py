import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest

        If the amount of work before starting exceeds the maximum number of
        threads allowed to the threadpool, only the maximum count will be
        started.
        