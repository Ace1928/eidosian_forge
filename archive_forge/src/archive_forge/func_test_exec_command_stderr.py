import os
import pytest
import sys
from tempfile import TemporaryFile
from numpy.distutils import exec_command
from numpy.distutils.exec_command import get_pythonexe
from numpy.testing import tempdir, assert_, assert_warns, IS_WASM
from io import StringIO
def test_exec_command_stderr():
    with redirect_stdout(TemporaryFile(mode='w+')):
        with redirect_stderr(StringIO()):
            with assert_warns(DeprecationWarning):
                exec_command.exec_command("cd '.'")
    if os.name == 'posix':
        with emulate_nonposix():
            with redirect_stdout(TemporaryFile()):
                with redirect_stderr(StringIO()):
                    with assert_warns(DeprecationWarning):
                        exec_command.exec_command("cd '.'")