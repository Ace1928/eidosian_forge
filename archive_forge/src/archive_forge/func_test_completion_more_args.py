import os
import shutil
import sys
import tempfile
import unittest
from os.path import join
from tempfile import TemporaryDirectory
from IPython.core.completerlib import magic_run_completer, module_completion, try_import
from IPython.testing.decorators import onlyif_unicode_paths
def test_completion_more_args(self):
    event = MockEvent(u'%run a.py ')
    match = set(magic_run_completer(None, event))
    self.assertEqual(match, set(self.files + self.dirs))