import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_privateBlobNoKeyObject(self):
    """
        Raises L{RuntimeError} if the underlying key object does not exists.
        """
    badKey = keys.Key(None)
    self.assertRaises(RuntimeError, badKey.privateBlob)