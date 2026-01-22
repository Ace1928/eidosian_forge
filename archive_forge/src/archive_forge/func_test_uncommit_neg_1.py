import os
from breezy import uncommit
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.errors import BoundBranchOutOfDate
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner, run_script
def test_uncommit_neg_1(self):
    wt = self.create_simple_tree()
    os.chdir('tree')
    out, err = self.run_bzr('uncommit -r -1', retcode=1)
    self.assertEqual('No revisions to uncommit.\n', out)