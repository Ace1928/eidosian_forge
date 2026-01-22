import os
from breezy import uncommit
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.errors import BoundBranchOutOfDate
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner, run_script
def test_uncommit_revision(self):
    wt = self.create_simple_tree()
    os.chdir('tree')
    out, err = self.run_bzr('uncommit -r1 --force')
    self.assertNotContainsRe(out, 'initial commit')
    self.assertContainsRe(out, 'second commit')
    self.assertEqual([b'a1'], wt.get_parent_ids())
    self.assertEqual(b'a1', wt.branch.last_revision())