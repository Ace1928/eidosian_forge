import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def test_content_filtering_applied_on_merge(self):
    source, path1, path2, path3, path4 = self.create_cf_tree_with_two_revisions(txt_reader=None, txt_writer=None, dir='source')
    if not source.supports_content_filtering():
        return
    self.assert_basis_content(b'Foo ROCKS!', source, path1)
    self.assertFileEqual(b'Foo ROCKS!', 'source/file1.txt')
    self.assert_basis_content(b'Foo Bin', source, path2)
    self.assert_basis_content(b'Hello World', source, path4)
    self.assertFileEqual(b'Hello World', 'source/file4.txt')
    self.patch_in_content_filter()
    self.run_bzr('branch -r1 source target')
    target = WorkingTree.open('target')
    self.assert_basis_content(b'Foo Txt', target, path1)
    self.assertFileEqual(b'fOO tXT', 'target/file1.txt')
    self.assertFileEqual(b'Foo Bin', 'target/file2.bin')
    self.assertFileEqual(b'bAR tXT', 'target/file3.txt')
    self.run_bzr('merge -d target source')
    self.assertFileEqual(b'fOO rocks!', 'target/file1.txt')
    self.assertFileEqual(b'hELLO wORLD', 'target/file4.txt')
    target.commit('merge file1.txt changes from source')
    self.assert_basis_content(b'Foo ROCKS!', target, path1)
    self.assert_basis_content(b'Hello World', target, path4)