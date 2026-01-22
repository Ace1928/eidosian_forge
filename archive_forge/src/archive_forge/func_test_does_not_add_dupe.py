import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def test_does_not_add_dupe(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('.bzrignore', b'myentry\n')])
    tree.add(['.bzrignore'])
    ignores.tree_ignores_add_patterns(tree, ['myentry'])
    self.assertPatternsEquals(['myentry'])