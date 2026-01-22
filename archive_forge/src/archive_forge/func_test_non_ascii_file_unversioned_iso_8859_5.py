import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_non_ascii_file_unversioned_iso_8859_5(self):
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('.')
    self.build_tree(['f'])
    tree.add(['f'])
    out, err = self.run_bzr_raw(['commit', '-m', 'Wrong filename', '§'], encoding='iso-8859-5', retcode=3)
    self.assertNotContainsString(err, b'\xc2\xa7')
    self.assertContainsRe(err, b'(?m)not versioned: "\xfd"$')