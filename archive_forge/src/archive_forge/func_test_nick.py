import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_nick(self):
    r = GitRepo.init(self.test_dir)
    dir = ControlDir.open(self.test_dir)
    dir.create_branch()
    output, error = self.run_bzr(['nick'])
    self.assertEqual('master\n', output)