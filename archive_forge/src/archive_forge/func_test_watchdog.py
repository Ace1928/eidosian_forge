from __future__ import absolute_import
import sys
import subprocess
import time
from twisted.trial.unittest import TestCase
from crochet._shutdown import (
from ..tests import crochet_directory
import threading, sys
from crochet._shutdown import register, _watchdog
def test_watchdog(self):
    """
        The watchdog thread exits when the thread it is watching exits, and
        calls its shutdown function.
        """
    done = []
    alive = True

    class FakeThread:

        def is_alive(self):
            return alive
    w = Watchdog(FakeThread(), lambda: done.append(True))
    w.start()
    time.sleep(0.2)
    self.assertTrue(w.is_alive())
    self.assertFalse(done)
    alive = False
    time.sleep(0.2)
    self.assertTrue(done)
    self.assertFalse(w.is_alive())