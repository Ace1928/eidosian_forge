import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
def chk_ops(a, b, expected_codes, n=3):
    s = self._PatienceSequenceMatcher(None, a, b)
    self.assertEqual(expected_codes, list(s.get_grouped_opcodes(n)))