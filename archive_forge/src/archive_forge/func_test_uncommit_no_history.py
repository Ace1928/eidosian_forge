import os
from breezy import uncommit
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.errors import BoundBranchOutOfDate
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner, run_script
def test_uncommit_no_history(self):
    wt = self.make_branch_and_tree('tree')
    out, err = self.run_bzr('uncommit --force', retcode=1)
    self.assertEqual('', err)
    self.assertEqual('No revisions to uncommit.\n', out)