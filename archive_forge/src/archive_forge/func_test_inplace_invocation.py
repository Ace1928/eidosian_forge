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
def test_inplace_invocation(self):
    """
    Test invocation for inplace format of a file
    """
    thisdir = os.path.realpath(os.path.dirname(__file__))
    infile_path = os.path.join(thisdir, 'testdata', 'test_in.cmake')
    expectfile_path = os.path.join(thisdir, 'testdata', 'test_out.cmake')
    ofd, tmpfile_path = tempfile.mkstemp(suffix='.txt', prefix='CMakeLists', dir=self.tempdir)
    os.close(ofd)
    shutil.copyfile(infile_path, tmpfile_path)
    os.chmod(tmpfile_path, 493)
    subprocess.check_call([sys.executable, '-Bm', 'cmakelang.format', '-i', tmpfile_path], cwd=self.tempdir, env=self.env)
    with io.open(os.path.join(tmpfile_path), 'r', encoding='utf8') as infile:
        actual_text = infile.read()
    with io.open(expectfile_path, 'r', encoding='utf8') as infile:
        expected_text = infile.read()
    delta_lines = list(difflib.unified_diff(expected_text.split('\n'), actual_text.split('\n')))
    if delta_lines:
        raise AssertionError('\n'.join(delta_lines[2:]))
    self.assertEqual(oct(os.stat(tmpfile_path).st_mode)[-3:], '755')