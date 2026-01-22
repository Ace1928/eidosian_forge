import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
@skipWithoutEd25519
def test_toOpenSSHEd25519_PEM_format(self):
    """
        L{keys.Key.toString} refuses to serialize an Ed25519 key in
        OpenSSH's old PEM format, as no encoding of Ed25519 is defined for
        that format.
        """
    key = keys.Key.fromString(keydata.privateEd25519_openssh_new)
    self.assertRaises(ValueError, key.toString, 'openssh', subtype='PEM')