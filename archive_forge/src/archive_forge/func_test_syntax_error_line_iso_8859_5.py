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
def test_syntax_error_line_iso_8859_5(self):
    """Syntax error on a iso-8859-5 line shows the line decoded"""
    text, raw = self._get_sample_text('iso-8859-5')
    textoutput = self._setup_external_case('import bad')
    self._write_module('bad', 'iso-8859-5', '# coding: iso-8859-5\n%% = 0 # %s\n' % text)
    textoutput = self._run_external_case()
    self.assertThat(textoutput, MatchesRegex(self._as_output(('.*%% = 0 # %s\n' + ' ' * self._error_on_character + '\\s*\\^\nSyntaxError:.*') % (text,)), re.MULTILINE | re.DOTALL))