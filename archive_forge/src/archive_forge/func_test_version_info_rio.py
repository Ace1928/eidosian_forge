import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_version_info_rio(self):
    output, error = self.run_bzr(['version-info', '--rio', 'gitr'])
    self.assertEqual(error, '')
    self.assertNotIn('revno:', output)