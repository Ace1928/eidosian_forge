from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_replace_property_with_invalid_value(self):
    """Values set on fake resource objects are validated."""
    self.launchpad.me = dict(name='foo')
    person = self.launchpad.me
    self.assertRaises(IntegrityError, setattr, person, 'name', 1)