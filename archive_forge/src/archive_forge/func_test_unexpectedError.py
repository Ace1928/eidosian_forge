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
def test_unexpectedError(self):
    """
        C{unexpectedError} reports an error processing a source file.
        """
    err = io.StringIO()
    reporter = Reporter(None, err)
    reporter.unexpectedError('source.py', 'error message')
    self.assertEqual('source.py: error message\n', err.getvalue())