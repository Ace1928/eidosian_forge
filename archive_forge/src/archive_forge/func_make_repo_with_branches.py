from .. import branch, controldir, tests, upgrade
from ..bzr import branch as bzrbranch
from ..bzr import workingtree, workingtree_4
def make_repo_with_branches(self):
    repo = self.make_repository('repo', shared=True, format=self.from_format)
    controldir.ControlDir.create_branch_convenience('repo/branch1', format=self.from_format)
    b2 = controldir.ControlDir.create_branch_convenience('repo/branch2', format=self.from_format)
    return repo.controldir