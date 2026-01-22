import gc
import threading
from weakref import ref
from twisted.internet.interfaces import IReactorThreads
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.threadable import isInIOThread
from twisted.python.threadpool import ThreadPool
from twisted.python.versions import Version

        The reactor registers itself as the I/O thread when it runs so that
        L{twisted.python.threadable.isInIOThread} returns C{False} if it is
        called in a different thread than the reactor is running in.
        