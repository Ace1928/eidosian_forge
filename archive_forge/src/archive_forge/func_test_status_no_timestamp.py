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
def test_status_no_timestamp(self):
    result = TimestampingStreamResult(LoggingStreamResult())
    result.status(test_id='A', test_status='B', test_tags='C', runnable='D', file_name='E', file_bytes=b'F', eof=True, mime_type='G', route_code='H')
    events = result.targets[0]._events
    self.assertThat(events, HasLength(1))
    self.assertThat(events[0], HasLength(11))
    self.assertEqual(('status', 'A', 'B', 'C', 'D', 'E', b'F', True, 'G', 'H'), events[0][:10])
    self.assertNotEqual(None, events[0][10])
    self.assertIsInstance(events[0][10], datetime.datetime)