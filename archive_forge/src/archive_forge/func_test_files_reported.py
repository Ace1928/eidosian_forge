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
def test_files_reported(self):
    tests = []
    result = StreamToDict(tests.append)
    result.startTestRun()
    result.status(file_name='some log.txt', file_bytes=_b('1234 log message'), eof=True, mime_type='text/plain; charset=utf8', test_id='foo.bar')
    result.status(file_name='another file', file_bytes=_b('Traceback...'), test_id='foo.bar')
    result.stopTestRun()
    self.assertThat(tests, HasLength(1))
    test = tests[0]
    self.assertEqual('foo.bar', test['id'])
    self.assertEqual('unknown', test['status'])
    details = test['details']
    self.assertEqual('1234 log message', details['some log.txt'].as_text())
    self.assertEqual(_b('Traceback...'), _b('').join(details['another file'].iter_bytes()))
    self.assertEqual('application/octet-stream', repr(details['another file'].content_type))