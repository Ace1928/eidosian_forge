from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_use_shared_to_standalone(self):
    repo = self.make_repository('repo', shared=True)
    branch = controldir.ControlDir.create_branch_convenience('repo/tree')
    self.assertNotEqual(branch.controldir.root_transport.base, branch.repository.controldir.root_transport.base)
    self.run_bzr('reconfigure --standalone', working_dir='repo/tree')
    tree = workingtree.WorkingTree.open('repo/tree')
    self.assertEqual(tree.controldir.root_transport.base, tree.branch.repository.controldir.root_transport.base)