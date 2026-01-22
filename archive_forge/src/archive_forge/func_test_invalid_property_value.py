from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_invalid_property_value(self):
    """
        The types of sample data values set on L{FakeLaunchpad} instances are
        validated against types defined in the WADL definition.
        """
    self.assertRaises(IntegrityError, setattr, self.launchpad, 'me', dict(name=102))