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
def test_changePassphraseCreateError(self) -> None:
    """
        L{changePassPhrase} doesn't modify the key file if an unexpected error
        happens when trying to create the key with the new passphrase.
        """
    filename = self.mktemp()
    FilePath(filename).setContent(privateRSA_openssh)

    def toString(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError('oops')
    self.patch(Key, 'toString', toString)
    error = self.assertRaises(SystemExit, changePassPhrase, {'filename': filename, 'newpass': 'newencrypt'})
    self.assertEqual('Could not change passphrase: oops', str(error))
    self.assertEqual(privateRSA_openssh, FilePath(filename).getContent())