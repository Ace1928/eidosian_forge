from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_standalone_to_use_shared(self):
    self.build_tree(['repo/'])
    tree = self.make_branch_and_tree('repo/tree')
    repo = self.make_repository('repo', shared=True)
    self.run_bzr('reconfigure --use-shared', working_dir='repo/tree')
    tree = workingtree.WorkingTree.open('repo/tree')
    self.assertNotEqual(tree.controldir.root_transport.base, tree.branch.repository.controldir.root_transport.base)