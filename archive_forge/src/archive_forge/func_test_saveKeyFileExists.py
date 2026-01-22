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
def test_saveKeyFileExists(self) -> None:
    """
        When the specified file exists, it will ask the user for confirmation
        before overwriting.
        """

    def mock_input(*args: object) -> list[str]:
        return ['n']
    base = FilePath(self.mktemp())
    base.makedirs()
    keyPath = base.child('custom_key').path
    self.patch(os.path, 'exists', lambda _: True)
    key = Key.fromString(privateRSA_openssh)
    options = {'filename': keyPath, 'no-passphrase': True, 'format': 'md5-hex'}
    self.assertRaises(SystemExit, _saveKey, key, options, mock_input)