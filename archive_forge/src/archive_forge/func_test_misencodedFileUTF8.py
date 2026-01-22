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
def test_misencodedFileUTF8(self):
    """
        If a source file contains bytes which cannot be decoded, this is
        reported on stderr.
        """
    SNOWMAN = chr(9731)
    source = ('# coding: ascii\nx = "%s"\n' % SNOWMAN).encode('utf-8')
    with self.makeTempFile(source) as sourcePath:
        self.assertHasErrors(sourcePath, [f"{sourcePath}:1:1: 'ascii' codec can't decode byte 0xe2 in position 21: ordinal not in range(128)\n"])