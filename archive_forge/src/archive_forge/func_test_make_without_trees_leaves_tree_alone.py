from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_make_without_trees_leaves_tree_alone(self):
    repo = self.make_repository('repo', shared=True)
    branch = controldir.ControlDir.create_branch_convenience('repo/branch')
    tree = workingtree.WorkingTree.open('repo/branch')
    self.build_tree(['repo/branch/foo'])
    tree.add('foo')
    self.run_bzr('reconfigure --with-no-trees --force', working_dir='repo/branch')
    self.assertPathExists('repo/branch/foo')
    tree = workingtree.WorkingTree.open('repo/branch')