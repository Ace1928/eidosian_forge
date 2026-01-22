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
@contextlib.contextmanager
def makeTempFile(self, content):
    """
        Make a temporary file containing C{content} and return a path to it.
        """
    fd, name = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'wb') as f:
            if not hasattr(content, 'decode'):
                content = content.encode('ascii')
            f.write(content)
        yield name
    finally:
        os.remove(name)