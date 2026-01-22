from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_replace_method(self):
    """Methods already set on fake resource objects can be replaced."""
    branch1 = dict(name='foo', bzr_identity='lp:~user/project/branch1')
    branch2 = dict(name='foo', bzr_identity='lp:~user/project/branch2')
    self.launchpad.branches = dict(getByUniqueName=lambda name: branch1)
    self.launchpad.branches.getByUniqueName = lambda name: branch2
    branch = self.launchpad.branches.getByUniqueName('foo')
    self.assertEqual('lp:~user/project/branch2', branch.bzr_identity)