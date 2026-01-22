from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_callable_object_no_response_representation(self):
    """
        If the WADL definition of a method does not include a response
        representation, then fake versions of that method just pass through
        the return value.
        """
    branch = dict(canBeDeleted=lambda: True)
    self.launchpad.branches = dict(getByUniqueName=lambda name: branch)
    branch = self.launchpad.branches.getByUniqueName('foo')
    self.assertTrue(branch.canBeDeleted())