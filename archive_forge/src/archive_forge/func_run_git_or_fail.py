import errno
import functools
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Tuple
from dulwich.tests import SkipTest, TestCase
from ...protocol import TCP_GIT_PORT
from ...repo import Repo
def run_git_or_fail(args, git_path=_DEFAULT_GIT, input=None, **popen_kwargs):
    """Run a git command, capture stdout/stderr, and fail if git fails."""
    if 'stderr' not in popen_kwargs:
        popen_kwargs['stderr'] = subprocess.STDOUT
    returncode, stdout, stderr = run_git(args, git_path=git_path, input=input, capture_stdout=True, capture_stderr=True, **popen_kwargs)
    if returncode != 0:
        raise AssertionError('git with args %r failed with %d: stdout=%r stderr=%r' % (args, returncode, stdout, stderr))
    return stdout