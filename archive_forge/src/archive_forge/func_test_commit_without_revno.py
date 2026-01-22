import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_commit_without_revno(self):
    repo = GitRepo.init(self.test_dir)
    output, error = self.run_bzr(['commit', '-Ocalculate_revnos=yes', '--unchanged', '-m', 'one'])
    self.assertContainsRe(error, 'Committed revision 1.')
    output, error = self.run_bzr(['commit', '-Ocalculate_revnos=no', '--unchanged', '-m', 'two'])
    self.assertNotContainsRe(error, 'Committed revision 2.')
    self.assertContainsRe(error, 'Committed revid .*.')