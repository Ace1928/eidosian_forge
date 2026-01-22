import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def test_content_filtering_applied_on_switch(self):
    source, path1, path2, path3, path4 = self.create_cf_tree_with_two_revisions(txt_reader=None, txt_writer=None, dir='branch-a')
    if not source.supports_content_filtering():
        return
    self.patch_in_content_filter()
    self.run_bzr('branch -r1 branch-a branch-b')
    self.run_bzr('checkout --lightweight branch-b checkout')
    self.assertFileEqual(b'fOO tXT', 'checkout/file1.txt')
    checkout_control_dir = ControlDir.open_containing('checkout')[0]
    switch(checkout_control_dir, source.branch)
    self.assertFileEqual(b'fOO rocks!', 'checkout/file1.txt')
    self.assertFileEqual(b'hELLO wORLD', 'checkout/file4.txt')