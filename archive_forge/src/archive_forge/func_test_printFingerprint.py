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
def test_printFingerprint(self) -> None:
    """
        L{printFingerprint} writes a line to standard out giving the number of
        bits of the key, its fingerprint, and the basename of the file from it
        was read.
        """
    filename = self.mktemp()
    FilePath(filename).setContent(publicRSA_openssh)
    printFingerprint({'filename': filename, 'format': 'md5-hex'})
    self.assertEqual(self.stdout.getvalue(), '2048 85:25:04:32:58:55:96:9f:57:ee:fb:a8:1a:ea:69:da temp\n')