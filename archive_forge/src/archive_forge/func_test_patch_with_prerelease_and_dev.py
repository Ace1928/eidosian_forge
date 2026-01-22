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
def test_patch_with_prerelease_and_dev(self):
    """
        `incremental.update package --patch` increments the patch version, and
        disregards any old prerelease/dev versions.
        """
    self.packagedir.child('_version.py').setContent(b'\nfrom incremental import Version\n__version__ = Version("inctestpkg", 1, 2, 3, release_candidate=1, dev=2)\n__all__ = ["__version__"]\n')
    out = []
    _run(u'inctestpkg', path=None, newversion=None, patch=True, rc=False, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
    self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 2, 4)\n__all__ = ["__version__"]\n')