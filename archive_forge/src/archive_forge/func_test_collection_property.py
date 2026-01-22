from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_collection_property(self):
    """
        Attributes that represent links to collections of other objects are
        set using a dict representing the collection.
        """
    bug = dict(id='1')
    branch = dict(linked_bugs=dict(entries=[bug]))
    self.launchpad.branches = dict(getByUniqueName=lambda name: branch)
    branch = self.launchpad.branches.getByUniqueName('foo')
    [bug] = list(branch.linked_bugs)
    self.assertEqual('1', bug.id)