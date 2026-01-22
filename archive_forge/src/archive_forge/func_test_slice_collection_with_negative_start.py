from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_slice_collection_with_negative_start(self):
    """
        A C{ValueError} is raised if a negative start value is used when
        slicing a sample collection set on a L{FakeLaunchpad} instance.
        """
    bug1 = dict(id='1', title='Bug #1')
    bug2 = dict(id='2', title='Bug #2')
    self.launchpad.bugs = dict(entries=[bug1, bug2])
    self.assertRaises(ValueError, lambda: self.launchpad.bugs[-1:])
    self.assertRaises(ValueError, lambda: self.launchpad.bugs[-1:2])