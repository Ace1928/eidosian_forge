import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_dataError(self):
    """
        The L{keys.Key.data} method raises RuntimeError for bad keys.
        """
    badKey = keys.Key(b'')
    self.assertRaises(RuntimeError, badKey.data)