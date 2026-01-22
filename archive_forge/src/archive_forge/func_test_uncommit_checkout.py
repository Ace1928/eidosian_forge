import os
from breezy import uncommit
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.errors import BoundBranchOutOfDate
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner, run_script
def test_uncommit_checkout(self):
    wt = self.create_simple_tree()
    checkout_tree = wt.branch.create_checkout('checkout')
    self.assertEqual([b'a2'], checkout_tree.get_parent_ids())
    os.chdir('checkout')
    out, err = self.run_bzr('uncommit --dry-run --force')
    self.assertContainsRe(out, 'Dry-run')
    self.assertNotContainsRe(out, 'initial commit')
    self.assertContainsRe(out, 'second commit')
    self.assertEqual([b'a2'], checkout_tree.get_parent_ids())
    out, err = self.run_bzr('uncommit --force')
    self.assertNotContainsRe(out, 'initial commit')
    self.assertContainsRe(out, 'second commit')
    self.assertEqual([b'a1'], checkout_tree.get_parent_ids())
    self.assertEqual(b'a1', wt.branch.last_revision())
    self.assertEqual([b'a2'], wt.get_parent_ids())