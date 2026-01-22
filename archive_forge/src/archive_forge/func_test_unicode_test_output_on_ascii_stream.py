import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
def test_unicode_test_output_on_ascii_stream(self):
    """Showing results should always succeed even on an ascii console"""

    class FailureWithUnicode(tests.TestCase):

        def test_log_unicode(self):
            self.log('☆')
            self.fail('Now print that log!')
    bio = BytesIO()
    out = TextIOWrapper(bio, 'ascii', 'backslashreplace')
    self.overrideAttr(osutils, 'get_terminal_encoding', lambda trace=False: 'ascii')
    self.run_test_runner(tests.TextTestRunner(stream=out), FailureWithUnicode('test_log_unicode'))
    out.flush()
    self.assertContainsRe(bio.getvalue(), b'(?:Text attachment: )?log(?:\n-+\n|: {{{)\\d+\\.\\d+  \\\\u2606(?:\n-+\n|}}}\n)')