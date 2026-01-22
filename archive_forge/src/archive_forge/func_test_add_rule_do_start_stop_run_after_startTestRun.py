import codecs
import datetime
import doctest
import io
from itertools import chain
from itertools import combinations
import os
import platform
from queue import Queue
import re
import shutil
import sys
import tempfile
import threading
from unittest import TestSuite
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.content_type import ContentType, UTF8_TEXT
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.tests.helpers import (
from testtools.testresult.doubles import (
from testtools.testresult.real import (
def test_add_rule_do_start_stop_run_after_startTestRun(self):
    nontest = LoggingStreamResult()
    router = StreamResultRouter()
    router.startTestRun()
    router.add_rule(nontest, 'test_id', test_id=None, do_start_stop_run=True)
    router.stopTestRun()
    self.assertEqual([('startTestRun',), ('stopTestRun',)], nontest._events)