import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
@skipWithoutEd25519
def test_reprPrivateEd25519(self):
    """
        The repr of a L{keys.Key} contains all the OpenSSH format for an
        Ed25519 private key.
        """
    self.assertEqual(repr(keys.Key(self.ed25519Obj)), dedent('                <Ed25519 Private Key (256 bits)\n                attr a:\n                \tf1:16:d1:15:4a:1e:15:0e:19:5e:19:46:b5:f2:44:\n                \t0d:b2:52:a0:ae:2a:6b:23:13:73:45:fd:40:d9:57:\n                \t7b:8b\n                attr k:\n                \t37:2f:25:da:8d:d4:a8:9a:78:7c:61:f0:98:01:c6:\n                \tf4:5e:6d:67:05:69:31:37:4c:69:0d:05:55:bb:c9:\n                \t44:58>'))