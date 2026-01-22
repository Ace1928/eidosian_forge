from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_set_undefined_property(self):
    """
        An L{IntegrityError} is raised if an attribute is set on a
        L{FakeLaunchpad} instance that isn't present in the WADL definition.
        """
    self.assertRaises(IntegrityError, setattr, self.launchpad, 'foo', 'bar')