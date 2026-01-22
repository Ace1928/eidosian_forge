from .... import tests
from .. import branch_mapper
from . import FastimportFeature
def test_git_to_bzr_for_trunk(self):
    m = branch_mapper.BranchMapper()
    for git, bzr in {b'refs/heads/trunk': 'git-trunk', b'refs/tags/trunk': 'git-trunk.tag', b'refs/remotes/origin/trunk': 'git-trunk.remote', b'refs/heads/git-trunk': 'git-git-trunk', b'refs/tags/git-trunk': 'git-git-trunk.tag', b'refs/remotes/origin/git-trunk': 'git-git-trunk.remote'}.items():
        self.assertEqual(m.git_to_bzr(git), bzr)