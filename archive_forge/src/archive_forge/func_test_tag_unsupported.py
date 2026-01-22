from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def test_tag_unsupported(self):
    tree = self.make_branch_and_tree('tree', format='dirstate')
    out, err = self.run_bzr('tag -d tree blah', retcode=3)
    self.assertEqual(out, '')
    self.assertContainsRe(err, "brz: ERROR: Tags not supported by BzrBranch5\\(.*\\/tree\\/\\); you may be able to use 'brz upgrade file:\\/\\/.*\\/tree\\/'.")