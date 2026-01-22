import contextlib
import functools
import io
import os
import shutil
import subprocess
import sys
import sysconfig
import setuptools
@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = (sys.stdout, sys.stderr)
    sys.stdout, sys.stderr = (io.StringIO(), io.StringIO())
    try:
        yield
    finally:
        sys.stdout, sys.stderr = (old_stdout, old_stderr)