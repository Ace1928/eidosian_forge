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
def test_printFingerprintHandleFileNotFound(self) -> None:
    """
        Ensure FileNotFoundError is handled for an invalid filename.
        """
    options = {'filename': '/foo/bar', 'format': 'md5-hex'}
    exc = self.assertRaises(SystemExit, printFingerprint, options)
    self.assertIn('could not be opened, please specify a file.', exc.args[0])