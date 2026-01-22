from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_report_changed(self):
    r = _mod_branch.PullResult()
    r.old_revid = b'old-revid'
    r.old_revno = 10
    r.new_revid = b'new-revid'
    r.new_revno = 20
    f = StringIO()
    r.report(f)
    self.assertEqual('Now on revision 20.\n', f.getvalue())