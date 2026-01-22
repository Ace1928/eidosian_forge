from __future__ import annotations
import getpass
import os
import subprocess
import sys
from io import StringIO
from typing import Callable
from typing_extensions import NoReturn
from twisted.conch.test.keydata import (
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_runBadKeytype(self) -> None:
    filename = self.mktemp()
    with self.assertRaises(subprocess.CalledProcessError):
        subprocess.check_call(['ckeygen', '-t', 'foo', '-f', filename])