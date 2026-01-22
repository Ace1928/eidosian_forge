from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_valid_append_revisions_only(self):
    self.assertEqual(None, self.config_stack.get('append_revisions_only'))
    self.check_append_revisions_only(None)
    self.check_append_revisions_only(False, 'False')
    self.check_append_revisions_only(True, 'True')
    self.check_append_revisions_only(False, 'false')
    self.check_append_revisions_only(True, 'true')