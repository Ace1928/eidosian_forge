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
def test_fileWithFlakes(self):
    """
        When a Python source file has warnings, the return code is non-zero
        and the warnings are printed to stdout.
        """
    with open(self.tempfilepath, 'wb') as fd:
        fd.write(b'import contraband\n')
    d = self.runPyflakes([self.tempfilepath])
    expected = UnusedImport(self.tempfilepath, Node(1), 'contraband')
    self.assertEqual(d, (f'{expected}{os.linesep}', '', 1))