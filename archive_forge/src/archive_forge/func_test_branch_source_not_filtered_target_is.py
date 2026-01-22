import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def test_branch_source_not_filtered_target_is(self):
    source, txt_path, bin_path = self.create_cf_tree(txt_reader=None, txt_writer=None, dir='source')
    if not source.supports_content_filtering():
        return
    self.assertFileEqual(b'Foo Txt', 'source/file1.txt')
    self.assert_basis_content(b'Foo Txt', source, txt_path)
    self.patch_in_content_filter()
    self.run_bzr('branch source target')
    target = WorkingTree.open('target')
    self.assertFileEqual(b'fOO tXT', 'target/file1.txt')
    changes = target.changes_from(source.basis_tree())
    self.assertFalse(changes.has_changed())