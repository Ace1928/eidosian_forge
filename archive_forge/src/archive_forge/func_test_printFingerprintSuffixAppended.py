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
def test_printFingerprintSuffixAppended(self) -> None:
    """
        L{printFingerprint} checks if the filename with the  '.pub' suffix
        exists in ~/.ssh.
        """
    filename = self.mktemp()
    FilePath(filename + '.pub').setContent(publicRSA_openssh)
    printFingerprint({'filename': filename, 'format': 'md5-hex'})
    self.assertEqual(self.stdout.getvalue(), '2048 85:25:04:32:58:55:96:9f:57:ee:fb:a8:1a:ea:69:da temp.pub\n')