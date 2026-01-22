from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
def test_get_root_id(self):
    tree = self.make_branch_and_tree('tree')
    if not tree.supports_file_ids:
        raise tests.TestNotApplicable('file ids not supported')
    root_id = tree.path2id('')
    if root_id is not None:
        self.assertIsInstance(root_id, bytes)