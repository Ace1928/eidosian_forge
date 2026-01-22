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
def test_stop_logging_on_exit(self):
    """
        setup() registers a reactor shutdown event that stops the logging
        thread.
        """
    observers = []
    reactor = FakeReactor()
    s = EventLoop(lambda: reactor, lambda f, *arg: None, lambda observer, setStdout=1: observers.append(observer))
    s.setup()
    self.addCleanup(observers[0].stop)
    self.assertIn(('after', 'shutdown', observers[0].stop), reactor.events)