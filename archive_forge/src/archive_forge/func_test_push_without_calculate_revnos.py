import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_push_without_calculate_revnos(self):
    self.run_bzr(['init', '--git', 'bla'])
    self.run_bzr(['init', '--git', 'foo'])
    self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo'])
    output, error = self.run_bzr(['push', '-Ocalculate_revnos=no', '-d', 'foo', 'bla'])
    self.assertEqual('', output)
    self.assertContainsRe(error, 'Pushed up to revision id git(.*).\n')