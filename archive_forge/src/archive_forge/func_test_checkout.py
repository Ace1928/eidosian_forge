import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_checkout(self):
    os.mkdir('gitbranch')
    GitRepo.init(os.path.join(self.test_dir, 'gitbranch'))
    os.chdir('gitbranch')
    builder = tests.GitBranchBuilder()
    builder.set_file(b'a', b'text for a\n', False)
    builder.commit(b'Joe Foo <joe@foo.com>', b'<The commit message>')
    builder.finish()
    os.chdir('..')
    output, error = self.run_bzr(['checkout', 'gitbranch', 'bzrbranch'])
    self.assertEqual(error, 'Fetching from Git to Bazaar repository. For better performance, fetch into a Git repository.\n')
    self.assertEqual(output, '')