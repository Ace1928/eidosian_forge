import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_cat_revision(self):
    self.simple_commit()
    output, error = self.run_bzr(['cat-revision', '-r-1'], retcode=3)
    self.assertContainsRe(error, 'brz: ERROR: Repository .* does not support access to raw revision texts')
    self.assertEqual(output, '')