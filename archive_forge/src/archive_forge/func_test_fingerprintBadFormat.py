import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fingerprintBadFormat(self):
    """
        A C{BadFingerPrintFormat} error is raised when unsupported
        formats are requested.
        """
    with self.assertRaises(keys.BadFingerPrintFormat) as em:
        keys.Key(self.rsaObj).fingerprint('sha256-base')
    self.assertEqual('Unsupported fingerprint format: sha256-base', em.exception.args[0])