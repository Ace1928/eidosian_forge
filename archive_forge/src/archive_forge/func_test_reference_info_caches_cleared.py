from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_reference_info_caches_cleared(self):
    branch = self.make_branch('branch')
    with branch.lock_write():
        branch.set_reference_info(b'file-id', 'location2', 'path2')
    doppelganger = _mod_branch.Branch.open('branch')
    doppelganger.set_reference_info(b'file-id', 'location3', 'path3')
    self.assertEqual(('location3', 'path3'), branch.get_reference_info(b'file-id'))