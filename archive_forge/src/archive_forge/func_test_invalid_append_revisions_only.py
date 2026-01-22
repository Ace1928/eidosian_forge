from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_invalid_append_revisions_only(self):
    """Ensure warning is noted on invalid settings"""
    self.warnings = []

    def warning(*args):
        self.warnings.append(args[0] % args[1:])
    self.overrideAttr(trace, 'warning', warning)
    self.check_append_revisions_only(None, 'not-a-bool')
    self.assertLength(1, self.warnings)
    self.assertEqual('Value "not-a-bool" is not valid for "append_revisions_only"', self.warnings[0])