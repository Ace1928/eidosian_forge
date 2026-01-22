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
def test_goodFile(self):
    """
        When a Python source file is all good, the return code is zero and no
        messages are printed to either stdout or stderr.
        """
    open(self.tempfilepath, 'a').close()
    d = self.runPyflakes([self.tempfilepath])
    self.assertEqual(d, ('', '', 0))