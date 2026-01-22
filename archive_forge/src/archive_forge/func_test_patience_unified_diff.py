import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
def test_patience_unified_diff(self):
    txt_a = ['hello there\n', 'world\n', 'how are you today?\n']
    txt_b = ['hello there\n', 'how are you today?\n']
    unified_diff = patiencediff.unified_diff
    psm = self._PatienceSequenceMatcher
    self.assertEqual(['--- \n', '+++ \n', '@@ -1,3 +1,2 @@\n', ' hello there\n', '-world\n', ' how are you today?\n'], list(unified_diff(txt_a, txt_b, sequencematcher=psm)))
    txt_a = [x + '\n' for x in 'abcdefghijklmnop']
    txt_b = [x + '\n' for x in 'abcdefxydefghijklmnop']
    self.assertEqual(['--- \n', '+++ \n', '@@ -1,6 +1,11 @@\n', ' a\n', ' b\n', ' c\n', '+d\n', '+e\n', '+f\n', '+x\n', '+y\n', ' d\n', ' e\n', ' f\n'], list(unified_diff(txt_a, txt_b)))
    self.assertEqual(['--- \n', '+++ \n', '@@ -4,6 +4,11 @@\n', ' d\n', ' e\n', ' f\n', '+x\n', '+y\n', '+d\n', '+e\n', '+f\n', ' g\n', ' h\n', ' i\n'], list(unified_diff(txt_a, txt_b, sequencematcher=psm)))