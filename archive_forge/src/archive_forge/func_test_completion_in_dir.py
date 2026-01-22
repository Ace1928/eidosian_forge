import os
import shutil
import sys
import tempfile
import unittest
from os.path import join
from tempfile import TemporaryDirectory
from IPython.core.completerlib import magic_run_completer, module_completion, try_import
from IPython.testing.decorators import onlyif_unicode_paths
def test_completion_in_dir(self):
    event = MockEvent(u'%run a.py {}'.format(join(self.BASETESTDIR, 'a')))
    print(repr(event.line))
    match = set(magic_run_completer(None, event))
    self.assertEqual(match, {join(self.BASETESTDIR, f).replace('\\', '/') for f in (u'a.py', u'aao.py', u'aao.txt', u'adir/')})