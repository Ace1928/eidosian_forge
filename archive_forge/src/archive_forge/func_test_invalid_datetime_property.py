from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_invalid_datetime_property(self):
    """
        Only C{datetime} values can be set on L{FakeLaunchpad} instances for
        attributes that represent dates.
        """
    self.assertRaises(IntegrityError, setattr, self.launchpad, 'me', dict(date_created='now'))