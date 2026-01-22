from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_datetime_property(self):
    """
        Attributes that represent dates are set with C{datetime} instances.
        """
    now = datetime.utcnow()
    self.launchpad.me = dict(date_created=now)
    self.assertEqual(now, self.launchpad.me.date_created)