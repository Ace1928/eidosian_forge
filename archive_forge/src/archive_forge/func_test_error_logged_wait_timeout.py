from __future__ import absolute_import
import threading
import subprocess
import time
import gc
import sys
import weakref
import tempfile
import os
import inspect
from unittest import SkipTest
from twisted.trial.unittest import TestCase
from twisted.internet.defer import succeed, Deferred, fail, CancelledError
from twisted.python.failure import Failure
from twisted.python import threadable
from twisted.python.runtime import platform
from .._eventloop import (
from .test_setup import FakeReactor
from .. import (
from ..tests import crochet_directory
import os, threading, signal, time, sys
import crochet
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred, CancelledError
import crochet
from crochet import EventualResult
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
import crochet
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
def test_error_logged_wait_timeout(self):
    """
        If the result is an error and wait() was called but timed out, the
        error will be logged once the EventualResult is garbage-collected.
        """
    d = Deferred()
    dr = EventualResult(d, None)
    try:
        dr.wait(0)
    except TimeoutError:
        pass
    d.errback(ZeroDivisionError())
    del dr
    if sys.version_info[0] == 2:
        sys.exc_clear()
    gc.collect()
    excs = self.flushLoggedErrors(ZeroDivisionError)
    self.assertEqual(len(excs), 1)