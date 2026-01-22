from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_instantiate_with_credentials(self):
    """A L{FakeLaunchpad} can be instantiated with credentials."""
    credentials = object()
    launchpad = FakeLaunchpad(credentials, application=get_application())
    self.assertEqual(credentials, launchpad.credentials)