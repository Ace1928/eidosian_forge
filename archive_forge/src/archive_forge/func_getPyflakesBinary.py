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
def getPyflakesBinary(self):
    """
        Return the path to the pyflakes binary.
        """
    import pyflakes
    package_dir = os.path.dirname(pyflakes.__file__)
    return os.path.join(package_dir, '..', 'bin', 'pyflakes')