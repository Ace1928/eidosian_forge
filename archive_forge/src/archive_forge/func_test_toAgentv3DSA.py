import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_toAgentv3DSA(self):
    """
        L{keys.Key.toString} serializes a DSA key in Agent v3 format.
        """
    key = keys.Key.fromString(keydata.privateDSA_openssh)
    self.assertEqual(key.toString('agentv3'), keydata.privateDSA_agentv3)