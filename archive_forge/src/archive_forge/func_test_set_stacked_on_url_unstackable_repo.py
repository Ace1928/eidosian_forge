from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_set_stacked_on_url_unstackable_repo(self):
    repo = self.make_repository('a', format='dirstate-tags')
    control = repo.controldir
    branch = _mod_bzrbranch.BzrBranchFormat7().initialize(control)
    target = self.make_branch('b')
    self.assertRaises(errors.UnstackableRepositoryFormat, branch.set_stacked_on_url, target.base)