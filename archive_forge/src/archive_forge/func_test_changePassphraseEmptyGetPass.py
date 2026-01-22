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
def test_changePassphraseEmptyGetPass(self) -> None:
    """
        L{changePassPhrase} exits if no passphrase is specified for the
        C{getpass} call and the key is encrypted.
        """
    self.patch(getpass, 'getpass', makeGetpass(''))
    filename = self.mktemp()
    FilePath(filename).setContent(privateRSA_openssh_encrypted)
    error = self.assertRaises(SystemExit, changePassPhrase, {'filename': filename})
    self.assertEqual('Could not change passphrase: Passphrase must be provided for an encrypted key', str(error))
    self.assertEqual(privateRSA_openssh_encrypted, FilePath(filename).getContent())