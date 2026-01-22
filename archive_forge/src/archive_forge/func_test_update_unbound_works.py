from breezy import branch, errors
from breezy import revision as _mod_revision
from breezy import tests
from breezy.tests import per_branch
def test_update_unbound_works(self):
    b = self.make_branch('.')
    b.update()
    self.assertEqual(_mod_revision.NULL_REVISION, b.last_revision())