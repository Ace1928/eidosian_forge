import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_log_verbose(self):
    self.simple_commit()
    output, error = self.run_bzr(['log', '-v'])
    self.assertContainsRe(output, 'revno: 1')