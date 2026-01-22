import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fingerprintsha256(self):
    """
        fingerprint method generates key fingerprint in
        L{FingerprintFormats.SHA256-BASE64} format if explicitly specified.
        """
    self.assertEqual(keys.Key(self.rsaObj).fingerprint(keys.FingerprintFormats.SHA256_BASE64), 'FBTCOoknq0mHy+kpfnY9tDdcAJuWtCpuQMaV3EsvbUI=')
    self.assertEqual(keys.Key(self.dsaObj).fingerprint(keys.FingerprintFormats.SHA256_BASE64), 'Wz5o2YbKyxOEcJn1au/UaALSVruUzfz0vaLI1xiIGyY=')