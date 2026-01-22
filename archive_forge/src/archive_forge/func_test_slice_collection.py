from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_slice_collection(self):
    """
        Data for a sample collection set on a L{FakeLaunchpad} instance can be
        sliced if an C{entries} key is defined.
        """
    bug1 = dict(id='1', title='Bug #1')
    bug2 = dict(id='2', title='Bug #2')
    bug3 = dict(id='3', title='Bug #3')
    self.launchpad.bugs = dict(entries=[bug1, bug2, bug3])
    bugs = self.launchpad.bugs[1:3]
    self.assertEqual(2, len(bugs))
    self.assertEqual('2', bugs[0].id)
    self.assertEqual('3', bugs[1].id)