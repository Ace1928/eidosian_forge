import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_local_whoami(self):
    r = GitRepo.init('gitr', mkdir=True)
    self.build_tree_contents([('gitr/.git/config', '[user]\n  email = some@example.com\n  name = Test User\n')])
    out, err = self.run_bzr(['whoami', '-d', 'gitr'])
    self.assertEqual(out, 'Test User <some@example.com>\n')
    self.assertEqual(err, '')
    self.build_tree_contents([('gitr/.git/config', '[user]\n  email = some@example.com\n')])
    out, err = self.run_bzr(['whoami', '-d', 'gitr'])
    self.assertEqual(out, 'some@example.com\n')
    self.assertEqual(err, '')