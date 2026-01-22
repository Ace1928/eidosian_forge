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
def test_control_characters_in_failure_string(self):
    """Control characters in assertions should be escaped"""
    textoutput = self._test_external_case("self.fail('\\a\\a\\a')")
    self.expectFailure('Defense against the beeping horror unimplemented', self.assertNotIn, self._as_output('\x07\x07\x07'), textoutput)
    self.assertIn(self._as_output('���'), textoutput)