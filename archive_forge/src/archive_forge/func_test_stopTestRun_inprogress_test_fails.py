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
def test_stopTestRun_inprogress_test_fails(self):
    result = StreamSummary()
    result.startTestRun()
    result.status('foo', 'inprogress')
    result.stopTestRun()
    self.assertEqual(False, result.wasSuccessful())
    self.assertThat(result.errors, HasLength(1))
    self.assertEqual('foo', result.errors[0][0].id())
    self.assertEqual('Test did not complete', result.errors[0][1])
    result.startTestRun()
    result.status('foo', 'inprogress')
    result.status('foo', 'inprogress', route_code='A')
    result.status('foo', 'success', route_code='A')
    result.stopTestRun()
    self.assertEqual(False, result.wasSuccessful())