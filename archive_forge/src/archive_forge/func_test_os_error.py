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
def test_os_error(self):
    """Locale error messages from the OS shouldn't break anything"""
    textoutput = self._test_external_case(modulelevel='import os', testline="os.mkdir('/')")
    self.assertThat(textoutput, self._local_os_error_matcher())