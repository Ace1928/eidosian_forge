from breezy.tests import TestNotApplicable
from breezy.tests.per_tree import TestCaseWithTree
from breezy.workingtree import SettingFileIdUnsupported
def test_get_root_id_default(self):
    tree = self.make_tree_with_default_root_id()
    if not tree.supports_file_ids:
        raise TestNotApplicable('file ids not supported')
    with tree.lock_read():
        self.assertIsNot(None, tree.path2id(''))