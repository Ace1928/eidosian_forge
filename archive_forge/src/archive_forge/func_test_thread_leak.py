import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
def test_thread_leak(self):
    """Ensure a thread that outlives the running of a test is reported

        Uses a thread that blocks on an event, and is started by the inner
        test case. As the thread outlives the inner case's run, it should be
        detected as a leak, but the event is then set so that the thread can
        be safely joined in cleanup so it's not leaked for real.
        """
    event = threading.Event()
    thread = threading.Thread(name='Leaker', target=event.wait)

    class Test(tests.TestCase):

        def test_leak(self):
            thread.start()
    result = self.LeakRecordingResult()
    test = Test('test_leak')
    self.addCleanup(thread.join)
    self.addCleanup(event.set)
    result.startTestRun()
    test.run(result)
    result.stopTestRun()
    self.assertEqual(result._tests_leaking_threads_count, 1)
    self.assertEqual(result._first_thread_leaker_id, test.id())
    self.assertEqual(result.leaks, [(test, {thread})])
    self.assertContainsString(result.stream.getvalue(), 'leaking threads')