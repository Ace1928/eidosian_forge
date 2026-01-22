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
def test_stopTestRun_count_many(self):
    test = make_test()
    self.result.startTestRun()
    self.result.startTest(test)
    self.result.stopTest(test)
    self.result.startTest(test)
    self.result.stopTest(test)
    self.result.stream = io.StringIO()
    self.result.stopTestRun()
    self.assertThat(self.getvalue(), DocTestMatches('\nRan 2 tests in ...s\n...', doctest.ELLIPSIS))