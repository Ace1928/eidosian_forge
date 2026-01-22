from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_no_functions(self):
    rev = self.branch.last_revision()
    self.assertEqual(None, self.branch.automatic_tag_name(rev))