import os
from breezy import tests
def test_mkdir_parents_existing_unversioned_dir(self):
    tree = self.make_branch_and_tree('.')
    os.mkdir('somedir')
    self.run_bzr(['mkdir', '-p', 'somedir'])
    self.assertEqual(tree.kind('somedir'), 'directory')