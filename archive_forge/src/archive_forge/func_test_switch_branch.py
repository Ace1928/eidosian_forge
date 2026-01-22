import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_switch_branch(self):
    repo = GitRepo.init(self.test_dir)
    builder = tests.GitBranchBuilder()
    builder.set_branch(b'refs/heads/oldbranch')
    builder.set_file('a', b'text for a\n', False)
    builder.commit(b'Joe Foo <joe@foo.com>', '<The commit message>')
    builder.set_branch(b'refs/heads/newbranch')
    builder.reset()
    builder.set_file('a', b'text for new a\n', False)
    builder.commit(b'Joe Foo <joe@foo.com>', '<The commit message>')
    builder.finish()
    repo.refs.set_symbolic_ref(b'HEAD', b'refs/heads/newbranch')
    repo.reset_index()
    output, error = self.run_bzr('switch oldbranch')
    self.assertEqual(output, '')
    self.assertTrue(error.startswith('Updated to revision 1.\n'), error)
    self.assertFileEqual('text for a\n', 'a')
    tree = WorkingTree.open('.')
    with tree.lock_read():
        basis_tree = tree.basis_tree()
        with basis_tree.lock_read():
            self.assertEqual([], list(tree.iter_changes(basis_tree)))