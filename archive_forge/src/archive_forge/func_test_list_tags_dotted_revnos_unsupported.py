from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def test_list_tags_dotted_revnos_unsupported(self):

    class TrimmedBranch(bzrbranch.BzrBranch6):

        def revision_id_to_dotted_revno(self, revid):
            raise errors.UnsupportedOperation(self.revision_id_to_dotted_revno, self)

    class TrimmedBranchFormat(bzrbranch.BzrBranchFormat6):

        def _branch_class(self):
            return TrimmedBranch

        @classmethod
        def get_format_string(cls):
            return b'Trimmed Branch'
    _mod_branch.format_registry.register(TrimmedBranchFormat())
    self.addCleanup(_mod_branch.format_registry.remove, TrimmedBranchFormat())
    dir_format = bzrdir.BzrDirMetaFormat1()
    dir_format._branch_format = TrimmedBranchFormat()
    tree = self.make_branch_and_tree('branch', format=dir_format)
    self.assertFileEqual('Trimmed Branch', 'branch/.bzr/branch/format')
    rev1 = tree.commit('rev1')
    tree.branch.tags.set_tag('mytag', rev1)
    out, err = self.run_bzr('tags -d branch', encoding='utf-8')
    self.assertEqual(out, 'mytag                ?\n')