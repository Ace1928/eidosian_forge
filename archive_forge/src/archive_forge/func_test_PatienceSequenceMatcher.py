import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
def test_PatienceSequenceMatcher(self):
    try:
        from ._patiencediff_c import PatienceSequenceMatcher_c
    except ImportError:
        from ._patiencediff_py import PatienceSequenceMatcher_py
        self.assertIs(PatienceSequenceMatcher_py, patiencediff.PatienceSequenceMatcher)
    else:
        self.assertIs(PatienceSequenceMatcher_c, patiencediff.PatienceSequenceMatcher)