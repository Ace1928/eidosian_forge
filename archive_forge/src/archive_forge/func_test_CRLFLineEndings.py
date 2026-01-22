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
def test_CRLFLineEndings(self):
    """
        Source files with Windows CR LF line endings are parsed successfully.
        """
    with self.makeTempFile('x = 42\r\n') as sourcePath:
        self.assertHasErrors(sourcePath, [])