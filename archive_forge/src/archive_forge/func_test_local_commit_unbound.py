import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_local_commit_unbound(self):
    self.make_branch_and_tree('.')
    out, err = self.run_bzr('commit --local', retcode=3)
    self.assertEqualDiff('', out)
    self.assertEqualDiff('brz: ERROR: Cannot perform local-only commits on unbound branches.\n', err)