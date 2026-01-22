from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_top_level_collection_with_invalid_entries(self):
    """
        Sample data for each entry in a collection is validated when it's set
        on a L{FakeLaunchpad} instance.
        """
    bug = dict(foo='bar')
    self.assertRaises(IntegrityError, setattr, self.launchpad, 'bugs', dict(entries=[bug]))