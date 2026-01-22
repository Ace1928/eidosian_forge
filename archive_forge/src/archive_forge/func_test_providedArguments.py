import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_providedArguments(self):
    """
        L{keys._getPersistentRSAKey} will put the key in
        C{directory}/C{filename}, with the key length of C{keySize}.
        """
    tempDir = FilePath(self.mktemp())
    keyFile = tempDir.child('mykey.pem')
    key = keys._getPersistentRSAKey(keyFile, keySize=1024)
    self.assertEqual(key.size(), 1024)
    self.assertTrue(keyFile.exists())