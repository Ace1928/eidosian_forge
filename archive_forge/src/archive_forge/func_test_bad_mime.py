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
def test_bad_mime(self):
    tests = []
    result = StreamToDict(tests.append)
    result.startTestRun()
    result.status(file_name='file', file_bytes=b'a', mime_type='text/plain; charset=utf8, language=python', test_id='id')
    result.stopTestRun()
    self.assertThat(tests, HasLength(1))
    test = tests[0]
    self.assertEqual('id', test['id'])
    details = test['details']
    self.assertEqual('a', details['file'].as_text())
    self.assertEqual('text/plain; charset="utf8"', repr(details['file'].content_type))