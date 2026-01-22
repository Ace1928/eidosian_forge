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
def test_local_tags_dont_leak(self):
    [result], events = self.make_results(1)
    a, b = (PlaceHolder('a'), PlaceHolder('b'))
    result.time(1)
    result.startTest(a)
    result.tags({'foo'}, set())
    result.time(2)
    result.addSuccess(a)
    result.stopTest(a)
    result.time(3)
    result.startTest(b)
    result.time(4)
    result.addSuccess(b)
    result.stopTest(b)
    self.assertEqual([('time', 1), ('startTest', a), ('time', 2), ('tags', {'foo'}, set()), ('addSuccess', a), ('stopTest', a), ('time', 3), ('startTest', b), ('time', 4), ('addSuccess', b), ('stopTest', b)], events)