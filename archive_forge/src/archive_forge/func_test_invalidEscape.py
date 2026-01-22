import contextlib
import io
import os
import sys
import shutil
import subprocess
import tempfile
from pyflakes.checker import PYPY
from pyflakes.messages import UnusedImport
from pyflakes.reporter import Reporter
from pyflakes.api import (
from pyflakes.test.harness import TestCase, skipIf
def test_invalidEscape(self):
    """
        The invalid escape syntax raises ValueError in Python 2
        """
    with self.makeTempFile("foo = '\\xyz'") as sourcePath:
        position_end = 1
        if PYPY and sys.version_info >= (3, 9):
            column = 7
        elif PYPY:
            column = 6
        elif (3, 9) <= sys.version_info < (3, 12):
            column = 13
        else:
            column = 7
        last_line = '%s^\n' % (' ' * (column - 1))
        decoding_error = "%s:1:%d: (unicode error) 'unicodeescape' codec can't decode bytes in position 0-%d: truncated \\xXX escape\nfoo = '\\xyz'\n%s" % (sourcePath, column, position_end, last_line)
        self.assertHasErrors(sourcePath, [decoding_error])