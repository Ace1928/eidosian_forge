from breezy import branch, controldir, errors, tests
from breezy.tests import script
def check_revno(self, val, loc='.'):
    self.assertEqual(val, controldir.ControlDir.open(loc).open_branch().last_revision_info()[0])