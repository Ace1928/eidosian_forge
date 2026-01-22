from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_repr_collection(self):
    """A custom C{__repr__} is provided for L{FakeCollection}s."""
    branches = dict(total_size='test-branch')
    self.launchpad.me = dict(getBranches=lambda statuses: branches)
    branches = self.launchpad.me.getBranches([])
    obj_id = hex(id(branches))
    self.assertEqual('<FakeCollection branch-page-resource object at %s>' % obj_id, repr(branches))