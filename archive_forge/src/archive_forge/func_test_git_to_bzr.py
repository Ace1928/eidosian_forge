from .... import tests
from .. import branch_mapper
from . import FastimportFeature
def test_git_to_bzr(self):
    m = branch_mapper.BranchMapper()
    for git, bzr in {b'refs/heads/master': 'trunk', b'refs/heads/foo': 'foo', b'refs/tags/master': 'trunk.tag', b'refs/tags/foo': 'foo.tag', b'refs/remotes/origin/master': 'trunk.remote', b'refs/remotes/origin/foo': 'foo.remote'}.items():
        self.assertEqual(m.git_to_bzr(git), bzr)