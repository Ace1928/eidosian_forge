import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def test_new_file(self):
    tree = self.make_branch_and_tree('.')
    ignores.tree_ignores_add_patterns(tree, ['myentry'])
    self.assertTrue(tree.has_filename('.bzrignore'))
    self.assertPatternsEquals(['myentry'])