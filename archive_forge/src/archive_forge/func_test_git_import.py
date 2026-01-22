import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_git_import(self):
    r = GitRepo.init('a', mkdir=True)
    self.build_tree(['a/file'])
    r.stage('file')
    r.do_commit(ref=b'refs/heads/abranch', committer=b'Joe <joe@example.com>', message=b'Dummy')
    r.do_commit(ref=b'refs/heads/bbranch', committer=b'Joe <joe@example.com>', message=b'Dummy')
    self.run_bzr(['git-import', '--colocated', 'a', 'b'])
    self.assertEqual({'.bzr'}, set(os.listdir('b')))
    self.assertEqual({'abranch', 'bbranch'}, set(ControlDir.open('b').branch_names()))