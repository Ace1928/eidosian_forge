import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
def test_recurse_matches(self):
    try:
        from ._patiencediff_c import recurse_matches_c
    except ImportError:
        from ._patiencediff_py import recurse_matches_py
        self.assertIs(recurse_matches_py, patiencediff.recurse_matches)
    else:
        self.assertIs(recurse_matches_c, patiencediff.recurse_matches)