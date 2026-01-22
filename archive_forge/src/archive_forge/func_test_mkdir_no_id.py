from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import SettingFileIdUnsupported
def test_mkdir_no_id(self):
    t = self.make_branch_and_tree('t1')
    t.lock_write()
    self.addCleanup(t.unlock)
    file_id = t.mkdir('path')
    self.assertEqual('directory', t.kind('path'))