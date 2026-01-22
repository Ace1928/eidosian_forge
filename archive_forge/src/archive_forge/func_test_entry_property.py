from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_entry_property(self):
    """
        Attributes that represent links to other objects are set using a
        dict representing the object.
        """
    bug = dict(owner=dict(name='test-person'))
    self.launchpad.bugs = dict(entries=[bug])
    bug = self.launchpad.bugs[0]
    self.assertEqual('test-person', bug.owner.name)