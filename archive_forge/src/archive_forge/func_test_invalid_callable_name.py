from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_invalid_callable_name(self):
    """
        An L{IntegrityError} is raised if a method is defined on a resource
        that doesn't match a method defined in the WADL definition.
        """
    self.assertRaises(IntegrityError, setattr, self.launchpad, 'me', dict(foo=lambda: None))