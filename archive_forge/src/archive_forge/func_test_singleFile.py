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
def test_singleFile(self):
    """
        If the directory contains one Python file, C{iterSourceCode} will find
        it.
        """
    childpath = self.makeEmptyFile('foo.py')
    self.assertEqual(list(iterSourceCode([self.tempdir])), [childpath])