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
def test_printFingerprintsha256(self) -> None:
    """
        L{printFigerprint} will print key fingerprint in
        L{FingerprintFormats.SHA256-BASE64} format if explicitly specified.
        """
    filename = self.mktemp()
    FilePath(filename).setContent(publicRSA_openssh)
    printFingerprint({'filename': filename, 'format': 'sha256-base64'})
    self.assertEqual(self.stdout.getvalue(), '2048 FBTCOoknq0mHy+kpfnY9tDdcAJuWtCpuQMaV3EsvbUI= temp\n')