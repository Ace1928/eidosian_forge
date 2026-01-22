import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromNewerOpenSSH(self):
    """
        Newer versions of OpenSSH generate encrypted keys which have a longer
        IV than the older versions.  These newer keys are also loaded.
        """
    key = keys.Key.fromString(keydata.privateRSA_openssh_encrypted_aes, passphrase=b'testxp')
    self.assertEqual(key.type(), 'RSA')
    key2 = keys.Key.fromString(keydata.privateRSA_openssh_encrypted_aes + b'\n', passphrase=b'testxp')
    self.assertEqual(key, key2)