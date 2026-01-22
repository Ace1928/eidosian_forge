import unittest as testm
from sys import version_info
import scrypt
def test_py3_hash_returns_bytes(self):
    """Test Py3 hash return bytes."""
    h = scrypt.hash(self.input, self.password)
    self.assertTrue(isinstance(h, bytes))