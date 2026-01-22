from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_instantiate_with_everything(self):
    """
        L{FakeLaunchpad} takes the same parameters as L{Launchpad} during
        instantiation, with the addition of an C{application} parameter.  The
        optional parameters are discarded when the object is instantiated.
        """
    credentials = object()
    launchpad = FakeLaunchpad(credentials, service_root=None, cache=None, timeout=None, proxy_info=None, application=get_application())
    self.assertEqual(credentials, launchpad.credentials)