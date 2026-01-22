from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def test_merge_without_commit_does_not_propagate_tags_to_master(self):
    """'brz merge' alone does not propagate tags to a master branch.

        (If the user runs 'brz commit', then that is when the tags from the
        merge are propagated.)
        """
    master, child = self.make_master_and_checkout()
    fork = self.make_fork(master)
    fork.tags.set_tag('new-tag', fork.last_revision())
    self.run_bzr(['merge', '../fork'], working_dir='child')
    self.assertEqual({}, master.tags.get_tag_dict())