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
def test_add_rule_route_code_consume_True(self):
    fallback = LoggingStreamResult()
    target = LoggingStreamResult()
    router = StreamResultRouter(fallback)
    router.add_rule(target, 'route_code_prefix', route_prefix='0', consume_route=True)
    router.status(test_id='foo', route_code='0')
    router.status(test_id='foo', route_code='0/1')
    router.status(test_id='foo', route_code='1')
    self.assertEqual([('status', 'foo', None, None, True, None, None, False, None, None, None), ('status', 'foo', None, None, True, None, None, False, None, '1', None)], target._events)
    self.assertEqual([('status', 'foo', None, None, True, None, None, False, None, '1', None)], fallback._events)