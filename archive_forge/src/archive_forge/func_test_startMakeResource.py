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
def test_startMakeResource(self):
    log = LoggingStreamResult()
    result = ResourcedToStreamDecorator(log)
    timestamp = datetime.datetime.utcfromtimestamp(3.476)
    result.startTestRun()
    result.time(timestamp)
    resource = testresources.TestResourceManager()
    result.startMakeResource(resource)
    [_, event] = log._events
    self.assertEqual('testresources.TestResourceManager.make', event.test_id)
    self.assertEqual('inprogress', event.test_status)
    self.assertFalse(event.runnable)
    self.assertEqual(timestamp, event.timestamp)