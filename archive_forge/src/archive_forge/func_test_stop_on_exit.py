from __future__ import absolute_import
import threading
import warnings
import subprocess
import sys
from unittest import SkipTest, TestCase
import twisted
from twisted.python.log import PythonLoggingObserver
from twisted.python import log
from twisted.python.runtime import platform
from twisted.internet.task import Clock
from .._eventloop import EventLoop, ThreadLogObserver, _store
from ..tests import crochet_directory
import sys
import crochet
import sys
from logging import StreamHandler, Formatter, getLogger, DEBUG
import crochet
from twisted.python import log
from twisted.logger import Logger
import time
def test_stop_on_exit(self):
    """
        setup() registers an exit handler that stops the reactor, and an exit
        handler that logs stashed EventualResults.
        """
    atexit = []
    reactor = FakeReactor()
    s = EventLoop(lambda: reactor, lambda f, *args: atexit.append((f, args)))
    s.setup()
    self.assertEqual(len(atexit), 2)
    self.assertFalse(reactor.stopping)
    f, args = atexit[0]
    self.assertEqual(f, reactor.callFromThread)
    self.assertEqual(args, (reactor.stop,))
    f(*args)
    self.assertTrue(reactor.stopping)
    f, args = atexit[1]
    self.assertEqual(f, _store.log_errors)
    self.assertEqual(args, ())
    f(*args)