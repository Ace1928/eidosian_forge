from __future__ import division, absolute_import
import sys
import os
import datetime
from twisted.python.filepath import FilePath
from twisted.python.compat import NativeStringIO
from twisted.trial.unittest import TestCase
from incremental.update import _run, run
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
def test_insufficient_args(self):
    """
        Calling run() with no args will cause it to print help.
        """
    stringio = NativeStringIO()
    self.patch(sys, 'stdout', stringio)
    self.patch(os, 'getcwd', self.getcwd)
    self.patch(datetime, 'date', self.date)
    with self.assertRaises(SystemExit) as e:
        run(['inctestpkg', '--rc'])
    self.assertEqual(e.exception.args[0], 0)
    self.assertIn('Updating codebase', stringio.getvalue())
    self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 16, 8, 0, release_candidate=1)\n__all__ = ["__version__"]\n')
    self.assertEqual(self.packagedir.child('__init__.py').getContent(), b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", 16, 8, 0, release_candidate=1).short()\nnext_released_version = "inctestpkg 16.8.0.rc1"\n')