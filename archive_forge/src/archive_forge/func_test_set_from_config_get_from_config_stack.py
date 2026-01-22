from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_set_from_config_get_from_config_stack(self):
    self.branch.lock_write()
    self.addCleanup(self.branch.unlock)
    self.branch.get_config().set_user_option('foo', 'bar')
    result = self.branch.get_config_stack().get('foo')
    self.expectFailure('BranchStack uses cache after set_user_option', self.assertEqual, 'bar', result)