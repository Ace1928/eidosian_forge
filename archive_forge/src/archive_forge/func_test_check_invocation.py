from __future__ import unicode_literals
import contextlib
import difflib
import io
import os
import shutil
import subprocess
import sys
import unittest
import tempfile
def test_check_invocation(self):
    """
    Test invocation for --check of a file
    """
    thisdir = os.path.realpath(os.path.dirname(__file__))
    unformatted_path = os.path.join(thisdir, 'testdata', 'test_in.cmake')
    formatted_path = os.path.join(thisdir, 'testdata', 'test_out.cmake')
    with open('/dev/null', 'wb') as devnull:
        statuscode = subprocess.call([sys.executable, '-Bm', 'cmakelang.format', '--check', unformatted_path], env=self.env, stderr=devnull)
    self.assertEqual(1, statuscode)
    statuscode = subprocess.call([sys.executable, '-Bm', 'cmakelang.format', '--check', formatted_path], env=self.env)
    self.assertEqual(0, statuscode)