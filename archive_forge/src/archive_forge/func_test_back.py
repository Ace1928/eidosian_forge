import tempfile
import unittest
from pathlib import Path
from bpython.config import getpreferredencoding
from bpython.history import History
def test_back(self):
    self.assertEqual(self.history.back(), '#999')
    self.assertNotEqual(self.history.back(), '#999')
    self.assertEqual(self.history.back(), '#997')
    for x in range(997):
        self.history.back()
    self.assertEqual(self.history.back(), '#0')