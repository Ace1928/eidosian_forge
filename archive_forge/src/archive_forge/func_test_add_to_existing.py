import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def test_add_to_existing(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('.bzrignore', b'myentry1\n')])
    tree.add(['.bzrignore'])
    ignores.tree_ignores_add_patterns(tree, ['myentry2', 'foo'])
    self.assertPatternsEquals(['myentry1', 'myentry2', 'foo'])