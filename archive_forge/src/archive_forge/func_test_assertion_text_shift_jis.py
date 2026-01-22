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
def test_assertion_text_shift_jis(self):
    """A terminal raw backslash in an encoded string is weird but fine"""
    example_text = '十'
    textoutput = self._test_external_case(coding='shift_jis', testline="self.fail('%s')" % example_text)
    output_text = example_text
    self.assertIn(self._as_output('AssertionError: %s' % output_text), textoutput)