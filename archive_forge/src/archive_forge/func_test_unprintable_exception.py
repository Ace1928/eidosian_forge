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
def test_unprintable_exception(self):
    """A totally useless exception instance still prints something"""
    exception_class = 'class UnprintableError(Exception):\n    def __str__(self):\n        raise RuntimeError\n    def __unicode__(self):\n        raise RuntimeError\n    def __repr__(self):\n        raise RuntimeError\n'
    if sys.version_info >= (3, 11):
        expected = 'UnprintableError: <exception str() failed>\n'
    else:
        expected = 'UnprintableError: <unprintable UnprintableError object>\n'
    textoutput = self._test_external_case(modulelevel=exception_class, testline='raise UnprintableError')
    self.assertIn(self._as_output(expected), textoutput)