import unittest as testm
from sys import version_info
import scrypt
def test_py3_encrypt_returns_bytes(self):
    """Test Py3 encrypt return bytes."""
    s = scrypt.encrypt(self.input, self.password, 0.1)
    self.assertTrue(isinstance(s, bytes))