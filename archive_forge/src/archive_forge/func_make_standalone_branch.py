from .. import branch, controldir, tests, upgrade
from ..bzr import branch as bzrbranch
from ..bzr import workingtree, workingtree_4
def make_standalone_branch(self):
    wt = self.make_branch_and_tree('branch1', format=self.from_format)
    return wt.controldir