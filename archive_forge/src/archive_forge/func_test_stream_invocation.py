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
def test_stream_invocation(self):
    """
    Test invocation with stdin as the infile and stdout as the outifle
    """
    thisdir = os.path.realpath(os.path.dirname(__file__))
    infile_path = os.path.join(thisdir, 'testdata', 'test_in.cmake')
    expectfile_path = os.path.join(thisdir, 'testdata', 'test_out.cmake')
    stdinpipe = os.pipe()
    stdoutpipe = os.pipe()

    def preexec():
        os.close(stdinpipe[1])
        os.close(stdoutpipe[0])
    proc = subprocess.Popen([sys.executable, '-Bm', 'cmakelang.format', '-'], stdin=stdinpipe[0], stdout=stdoutpipe[1], cwd=self.tempdir, env=self.env, preexec_fn=preexec)
    os.close(stdinpipe[0])
    os.close(stdoutpipe[1])
    with io.open(infile_path, 'r', encoding='utf-8') as infile:
        with io.open(stdinpipe[1], 'w', encoding='utf-8') as outfile:
            for line in infile:
                outfile.write(line)
    with io.open(stdoutpipe[0], 'r', encoding='utf-8') as infile:
        actual_text = infile.read()
    proc.wait()
    with io.open(expectfile_path, 'r', encoding='utf8') as infile:
        expected_text = infile.read()
    delta_lines = list(difflib.unified_diff(expected_text.split('\n'), actual_text.split('\n')))
    if delta_lines:
        raise AssertionError('\n'.join(delta_lines[2:]))