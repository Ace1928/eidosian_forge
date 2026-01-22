import os
import re
import unittest
from breezy import bzr, config, controldir, errors, osutils, repository, tests
from breezy.bzr.groupcompress_repo import RepositoryFormat2a
@unittest.skip('encoding when LANG=C is currently borked')
def test_utf8_default_fs_enc(self):
    """In the C locale brz treats a posix filesystem as UTF-8 encoded"""
    if os.name != 'posix':
        raise tests.TestNotApplicable('Needs system beholden to C locales')
    out, err = self.run_brz_subprocess(['init', 'file:%C2%A7'], env_changes={'LANG': 'C', 'LC_ALL': 'C'})
    self.assertContainsRe(out, b'^Created a standalone tree .*$')