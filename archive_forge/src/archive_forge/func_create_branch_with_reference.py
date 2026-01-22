from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def create_branch_with_reference(self):
    branch = self.make_branch('branch')
    branch._set_all_reference_info({'path': ('location', b'file-id')})
    return branch