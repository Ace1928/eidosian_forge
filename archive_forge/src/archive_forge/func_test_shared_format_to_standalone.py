from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_shared_format_to_standalone(self, format=None):
    repo = self.make_repository('repo', shared=True, format=format)
    branch = controldir.ControlDir.create_branch_convenience('repo/tree')
    self.assertNotEqual(branch.controldir.root_transport.base, branch.repository.controldir.root_transport.base)
    tree = workingtree.WorkingTree.open('repo/tree')
    self.build_tree_contents([('repo/tree/file', b'foo\n')])
    tree.add(['file'])
    tree.commit('added file')
    self.run_bzr('reconfigure --standalone', working_dir='repo/tree')
    tree = workingtree.WorkingTree.open('repo/tree')
    self.build_tree_contents([('repo/tree/file', b'bar\n')])
    self.check_file_contents('repo/tree/file', b'bar\n')
    self.run_bzr('revert', working_dir='repo/tree')
    self.check_file_contents('repo/tree/file', b'foo\n')
    self.assertEqual(tree.controldir.root_transport.base, tree.branch.repository.controldir.root_transport.base)