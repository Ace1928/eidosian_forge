from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def test_commit_in_heavyweight_checkout_reports_tag_conflict(self):
    master, child = self.make_master_and_checkout()
    fork = self.make_fork(master)
    fork.tags.set_tag('new-tag', fork.last_revision())
    master_r1 = master.last_revision()
    master.tags.set_tag('new-tag', master_r1)
    script.run_script(self, '\n            $ cd child\n            $ brz merge ../fork\n            $ brz commit -m "Merge fork."\n            2>Committing to: .../master/\n            2>Conflicting tags in bound branch:\n            2>    new-tag\n            2>Committed revision 2.\n            ', null_output_matches_anything=True)
    self.assertEqual({'new-tag': fork.last_revision()}, child.branch.tags.get_tag_dict())
    self.assertEqual({'new-tag': master_r1}, master.tags.get_tag_dict())