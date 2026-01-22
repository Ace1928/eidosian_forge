import os
from ...transform import ROOT_PARENT, conflict_pass, resolve_conflicts, revert
from . import TestCaseWithTransport
def test_directory_exists(self):
    tree = self.make_branch_and_tree('.', format='git')
    tt = tree.transform()
    dir1 = tt.new_directory('dir', ROOT_PARENT)
    tt.new_file('name1', dir1, [b'content1'])
    dir2 = tt.new_directory('dir', ROOT_PARENT)
    tt.new_file('name2', dir2, [b'content2'])
    raw_conflicts = resolve_conflicts(tt, None, lambda t, c: conflict_pass(t, c))
    conflicts = tt.cook_conflicts(raw_conflicts)
    self.assertEqual([], list(conflicts))
    tt.apply()
    self.assertEqual({'name1', 'name2'}, set(os.listdir('dir')))