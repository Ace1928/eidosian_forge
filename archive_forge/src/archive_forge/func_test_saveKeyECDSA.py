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
def test_saveKeyECDSA(self) -> None:
    """
        L{_saveKey} writes the private and public parts of a key to two
        different files and writes a report of this to standard out.
        Test with ECDSA key.
        """
    base = FilePath(self.mktemp())
    base.makedirs()
    filename = base.child('id_ecdsa').path
    key = Key.fromString(privateECDSA_openssh)
    _saveKey(key, {'filename': filename, 'pass': 'passphrase', 'format': 'md5-hex'})
    self.assertEqual(self.stdout.getvalue(), 'Your identification has been saved in %s\nYour public key has been saved in %s.pub\nThe key fingerprint in <FingerprintFormats=MD5_HEX> is:\n1e:ab:83:a6:f2:04:22:99:7c:64:14:d2:ab:fa:f5:16\n' % (filename, filename))
    self.assertEqual(key.fromString(base.child('id_ecdsa').getContent(), None, 'passphrase'), key)
    self.assertEqual(Key.fromString(base.child('id_ecdsa.pub').getContent()), key.public())