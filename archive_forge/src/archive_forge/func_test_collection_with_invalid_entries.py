from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_collection_with_invalid_entries(self):
    """
        Sample data for each entry in a collection is validated when it's set
        on an attribute representing a link to a collection of objects.
        """
    bug = dict(foo='bar')
    branch = dict(linked_bugs=dict(entries=[bug]))
    self.launchpad.branches = dict(getByUniqueName=lambda name: branch)
    self.assertRaises(IntegrityError, self.launchpad.branches.getByUniqueName, 'foo')