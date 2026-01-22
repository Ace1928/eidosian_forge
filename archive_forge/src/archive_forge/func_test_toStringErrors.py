import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_toStringErrors(self):
    """
        L{keys.Key.toString} raises L{keys.BadKeyError} when passed an invalid
        format type.
        """
    self.assertRaises(keys.BadKeyError, keys.Key(self.rsaObj).toString, 'bad_type')